import torch
from collections import defaultdict
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import construct_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
from models.model_component.norm_type import *
from models.model_component.resnet_encoder import ResnetEncoder
from models.model_component.voxel_fusion_net import Vox_fusion_net
from models.model_component.vox2dimg_net import Vox_dimg_net
from models.model_component.DepthEstimation.Solo import CVT2d
from models.model_component.DepthEstimation.depth_decoder import DepthDecoder
from models.model_component.PoseEstimation.Posenet import IF_PoseNet

from models.model_component.depth_estimation_net import Depth_EstimationNet
import torch.distributed as dist
from models.backproject import BackprojectDepth,Project3D,SSIM
import time
from utils.logger import *
_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

class M2depth:
    def __init__(self, cfg, rank):
        # super(M2depth, self).__init__(cfg)
        self.read_config(cfg)
        self.log_path = os.path.join(self.log_dir, self.model_name)
        os.makedirs(os.path.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'models'), exist_ok=True)

        self.dataloaders = {}
        self.rank = rank
        torch.cuda.set_device(self.rank)
        # dist.init_process_group(backend='nccl')
        self.device = torch.device("cuda", self.rank)

        self.ssim = SSIM()
        self.prepare_dataset(cfg, rank)
        self.train_dataloader = self.dataloaders['train']
        self.parameters_to_train = []
        # self.models = self.prepare_model(cfg,rank)
        self.prepare_model(cfg,rank)
        self.backproject_depth = {}
        self.project_3d = {}
        self.prepare_comparison()

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)

        # self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)


    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def prepare_comparison(self):

        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size*6, h, w)
            # self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size*6, h, w)
            # self.project_3d[scale].to(self.device)


    def prepare_model(self, cfg, rank):
        self.models = {}
        self.models['encoder'] = ResnetEncoder(self.num_layers, self.weights_init, 1)
        self.models["encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["encoder"])
        self.models["encoder"] = (self.models["encoder"]).to(self.device)


        self.models["depth_estimation"] = Depth_EstimationNet(cfg,rank)
        self.models["depth_estimation"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth_estimation"])
        self.models["depth_estimation"] = (self.models["depth_estimation"]).to(self.device)



        self.models["Posenet_if"] = IF_PoseNet(cfg)
        self.models["Posenet_if"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["Posenet_if"])
        self.models["Posenet_if"] = (self.models["Posenet_if"]).to(self.device)

        self.parameters_to_train += list(self.models['encoder'].parameters())
        self.parameters_to_train += list(self.models["depth_estimation"].parameters())
        self.parameters_to_train += list(self.models["Posenet_if"].parameters())

        # for key in self.models.keys():
        #     self.models[key] = DDP(self.models[key], device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True, broadcast_buffers=False)
        #


        # DDP training
        # if self.ddp_enable == True:
        #     from torch.nn.parallel import DistributedDataParallel as DDP
        #     process_group = dist.new_group(list(range(self.world_size)))
        #     # set ddp configuration
        #     for k, v in self.models.items():
        #         # sync batchnorm
        #         v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v, process_group)
        #         # DDP enable
        #         self.models[k] = DDP(v, device_ids=[rank], broadcast_buffers=True)


        # models['vox_fusion'] = Vox_fusion_net(cfg, rank)
        # models["vox2dimg"] = Vox_dimg_net(cfg, rank)
        # self.fusion_level_start_dim = sum(self.fusion_level_list[self.fusion_level:])
        # models["conv2d"] = conv2d(self.fusion_level_start_dim, self.fusion_feat_in_dim, kernel_size=1,
        #                      padding_mode='reflect')
        # CVT_solo = []
        # for i in range(self.fusion_level + 1):
        #     CVT_solo.append(CVT2d(input_channel=self.fusion_level_list[i], downsample_ratio=2 ** (5 - 1 - i),
        #                                iter_num=self.CVT_iter_num))
        # models["CVY_solo"] = CVT_solo
        # num_ch_dec = [16, 32, 64, 128, 256]
        # models["depth_decoder"] =  DepthDecoder(self.fusion_level, self.fusion_level_list[:self.fusion_level + 1], num_ch_dec,
        #                             self.scales, use_skips=self.use_skips)

        # return models

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')

        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            self.set_eval_dataloader(cfg, rank)

            # if rank == 0:
            #     self.set_val_dataloader(cfg)

        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)


    def set_train_dataloader(self, cfg, rank):
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if self.ddp_enable:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas = self.world_size,
                rank=rank,
                shuffle=True
            )
            dataloader_opts['sampler'] = self.train_sampler

        self.dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_eval_dataloader(self, cfg, rank):
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        B, C, H, W = inputs[("color", 0, 0)][0].shape
        for scale in self.scales:
            disp = outputs[("disp", scale)]

            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)


            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # 注意，这里对于相机本身的外参并没有做特殊处理，应该是假设相机之间不会发生相对运动（s and t）
                cam_points = self.backproject_depth[source_scale](depth, inputs[('inv_K',0)].view(-1,4,4))
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K",0)].view(-1,4,4), T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)].view(-1,C,H,W),
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if self.automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

                if self.spatial:
                    T = inputs[('pose_spatial', frame_id)]

                    cam_points = self.backproject_depth[source_scale](outputs[("depth", 0, scale)], inputs[("inv_K", 0, source_scale)])

                    K_temp = inputs[("K", 0, source_scale)].clone().reshape(-1, 6, 4, 4)
                    if frame_id == 1:
                        K_temp = K_temp[:, [1, 2, 3, 4, 5, 0]]
                        K_temp = K_temp.reshape(-1, 4, 4)
                    elif frame_id == -1:
                        K_temp = K_temp[:, [5, 0, 1, 2, 3, 4]]
                        K_temp = K_temp.reshape(-1, 4, 4)
                    pix_coords = self.project_3d[source_scale](
                        cam_points, K_temp, T)

                    outputs[("sample_spatial", frame_id, scale)] = pix_coords

                    B, C, H, W = inputs[("color", 0, source_scale)][0].shape
                    inputs_temp = inputs[("color", 0, source_scale)].reshape(-1, 6, C, H, W)
                    if self.use_fix_mask:
                        inputs_mask = inputs["mask"].clone().reshape(-1, 6, 2, H, W)
                    if frame_id == 1:
                        inputs_temp = inputs_temp[:, [1, 2, 3, 4, 5, 0]]
                        inputs_temp = inputs_temp.reshape(B, C, H, W)
                        if self.use_fix_mask:
                            inputs_mask = inputs_mask[:, [1, 2, 3, 4, 5, 0]]
                            inputs_mask = inputs_mask.reshape(B, 2, H, W)
                    elif frame_id == -1:
                        inputs_temp = inputs_temp[:, [5, 0, 1, 2, 3, 4]]
                        inputs_temp = inputs_temp.reshape(B, C, H, W)
                        if self.use_fix_mask:
                            inputs_mask = inputs_mask[:, [5, 0, 1, 2, 3, 4]]
                            inputs_mask = inputs_mask.reshape(B, 2, H, W)

                    outputs[("color_spatial", frame_id, scale)] = F.grid_sample(
                        inputs_temp,
                        outputs[("sample_spatial", frame_id, scale)],
                        padding_mode="zeros", align_corners=True)

                    if self.use_fix_mask:
                        outputs[("color_spatial_mask", frame_id, scale)] = F.grid_sample(
                            inputs_mask[:, 0:1],
                            outputs[("sample_spatial", frame_id, scale)],
                            padding_mode="zeros", align_corners=True, mode='nearest').detach()
                    else:
                        outputs[("color_spatial_mask", frame_id, scale)] = F.grid_sample(
                            torch.ones(B, 1, H, W).cuda(),
                            outputs[("sample_spatial", frame_id, scale)],
                            padding_mode="zeros", align_corners=True, mode='nearest').detach()

            if self.use_sfm_spatial:
                outputs[("depth_match_spatial", scale)] = []
                inputs[("depth_match_spatial", scale)] = []

                for j in range(len(inputs["match_spatial"])):
                    pix_norm = inputs['match_spatial'][j][:, :2]
                    pix_norm[..., 0] /= self.opt.width_ori
                    pix_norm[..., 1] /= self.opt.height_ori
                    pix_norm = (pix_norm - 0.5) * 2

                    depth_billi = F.grid_sample(outputs[("depth", 0, scale)][j].unsqueeze(0),
                                                pix_norm.unsqueeze(1).unsqueeze(0), padding_mode="border")
                    depth_billi = depth_billi.squeeze()

                    compute_depth = inputs['match_spatial'][j][:, 2]
                    compute_angle = inputs['match_spatial'][j][:, 3]
                    distances1 = inputs['match_spatial'][j][:, 4]
                    distances2 = inputs['match_spatial'][j][:, 5]

                    triangulation_mask = (compute_depth > 0).float() * (compute_depth < 200).float() * (
                                compute_angle > 0.01).float() * (distances1 < self.opt.thr_dis).float() * (
                                                     distances2 < self.opt.thr_dis).float()

                    outputs[("depth_match_spatial", scale)].append(depth_billi[triangulation_mask == 1])
                    inputs[("depth_match_spatial", scale)].append(compute_depth[triangulation_mask == 1])

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        # print("repro_loss", l1_loss.size(),ssim_loss.size())
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0


        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            B, N, C, H, W = inputs[("color", 0, scale)].size()
            color = inputs[("color", 0, scale)].view(-1,C,H,W)
            B, N, C, H, W = inputs[("color", 0, source_scale)].size()
            target = inputs[("color", 0, source_scale)].view(-1,C,H,W)

            # print("loss: ",disp.size(), color.size(), target.size())

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                # print("loss pred: ", pred.size(), target.size())
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if self.automasking:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    B, N, C, H, W = inputs[("color", frame_id, source_scale)].size()
                    pred = inputs[("color", frame_id, source_scale)].view(-1,C,H,W)
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.height, self.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.use_fix_mask:
                reprojection_losses *= inputs["mask"]  # * output_mask

            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if self.automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001
                # identity_reprojection_loss += torch.randn(
                #     identity_reprojection_loss.shape) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            if self.use_sfm_spatial:
                depth_losses = []
                for j in range(len(inputs["match_spatial"])):
                    depth_loss = torch.abs(
                        outputs[("depth_match_spatial", scale)][j] - inputs[("depth_match_spatial", scale)][j]).mean()
                    depth_losses.append(depth_loss)
                loss += self.opt.match_spatial_weight * torch.stack(depth_losses).mean()

            if self.spatial:
                reprojection_losses_spatial = []
                spatial_mask = []
                target = inputs[("color", 0, source_scale)][0]

                for frame_id in self.frame_ids[1:]:
                    pred = outputs[("color_spatial", frame_id, scale)]

                    reprojection_losses_spatial.append(
                        outputs[("color_spatial_mask", frame_id, scale)] * self.compute_reprojection_loss(pred, target))

                reprojection_loss_spatial = torch.cat(reprojection_losses_spatial, 1)
                if self.use_fix_mask:
                    reprojection_loss_spatial *= inputs["mask"]

                loss += self.spatial_weight * reprojection_loss_spatial.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness_weight * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        # total_loss /= len(self.scales)
        losses["loss"] = total_loss
        return losses


    def process_batch(self,inputs):
        # print("len dataset:", len(self.dataloaders['train']),len(self.dataloaders['eval']))
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(self.rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(self.rank)
        # for key, ipt in inputs.items():
        #     if key not in _NO_DEVICE_KEYS:
        #         print(inputs[key].device)


        inputs[('K', 0)] = inputs[("K",0)].float()
        inputs[('inv_K', 0)] = inputs[("inv_K",0)].float()
        inputs['K',self.fusion_level+1] = inputs['K',self.fusion_level+1].float()
        inputs['inv_K',self.fusion_level+1] = inputs['inv_K',self.fusion_level+1].float()

        lev = self.fusion_level
        # img_encoder = self.models['encoder']
        sf_images_cur = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        sf_images_pre = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)

        packed_input = pack_cam_feat(sf_images_cur)
        packed_feats = self.models['encoder'](packed_input)
        outputs = self.models["depth_estimation"](inputs, packed_feats)
        print("pack_inputs is: ",packed_input.size())
        print("outputs keys is:", outputs.keys(), outputs[('disp', 0)].size())

        for frame_id in self.frame_ids[1:]:
            axisangle, translation = self.models["Posenet_if"](inputs, frame_id )
            outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(frame_id < 0))
            outputs[("axisangle", 0, frame_id)] = axisangle
            outputs[("translation", 0, frame_id)] = translation



        self.generate_images_pred(inputs, outputs)

        print("final output: ", outputs.keys())
        print("final shape: ", outputs[('disp', 2)].size(), outputs[('disp', 1)].size(), outputs[('disp', 0)].size()\
              , outputs[('cam_T_cam', 0, -1)].size() ,outputs[('axisangle', 0, -1)].size(), \
                outputs[('translation', 0, -1)].size(), outputs[('cam_T_cam', 0, 1)].size() ,outputs[('axisangle', 0, 1)].size(), \
              outputs[('translation', 0, 1)].size())

        losses = self.compute_losses(inputs, outputs)
        print("losses: ",losses)

        return outputs, losses

    def save_model(self):
        """Save model weights to disk
        """
        if self.rank == 0:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.step))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.module.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.height
                    to_save['width'] = self.width
                    # to_save['use_stereo'] = self.use_stereo
                torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)




    def validate(self, model):
        """
        This function validates models on validation dataset to monitor training process.
        """
        self.set_eval()
        inputs = next(self.val_iter)

        outputs, losses = self.process_batch(inputs)

        if 'depth' in inputs:
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, vis_scale=True)
            self.logger.print_perf(depth_eval_metric, 'metric')
            self.logger.print_perf(depth_eval_median, 'median')

        self.logger.log_tb('val', inputs, outputs, losses, self.step)
        del inputs, outputs, losses

        model.set_train()

    def run_per_epoch(self):
        torch.autograd.set_detect_anomaly(True)
        if self.rank == 0:
            print("Start Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.dataloaders['train']):
            before_op_time = time.time()
            self.model_optimizer.zero_grad()
            outputs, losses = self.process_batch(inputs)

            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                print(losses)

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

            if self.step % self.log_frequency == 0  and self.log_frequency > 0:
                self.save_model()
                # self.val()
                # if self.local_rank == 0:
                #     self.evaluation()
            self.step += 1
        # return 0

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        if self.rank == 0:
            samples_per_sec = self.batch_size / duration
            time_sofar = time.time() - self.start_time
            training_time_left = (
                                         self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                           " | loss: {:.5f} | time elapsed: {} | time left: {}"

            self.log_print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                               sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log_print(self, str):
        print(str)
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(str + '\n')

    def train(self):
        self.epoch = 0
        self.val_iter = iter(self.dataloaders["eval"])
        self.start_time = time.time()
        self.step = 1
        for self.epoch in range(self.num_epochs):
            self.dataloaders['train'].sampler.set_epoch(self.epoch)
            self.run_per_epoch()

        # return 0
    # def prossess_batch_1(self,inputs):
    #
    #     inputs['K',self.fusion_level+1] = inputs['K',self.fusion_level+1].float()
    #     inputs['inv_K',self.fusion_level+1] = inputs['inv_K',self.fusion_level+1].float()
    #
    #     lev = self.fusion_level
    #     # img_encoder = self.models['encoder']
    #
    #
    #
    #     sf_images_cur = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
    #     sf_images_pre = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
    #     sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
    #
    #     packed_input = pack_cam_feat(sf_images_cur)
    #     print("pack_inputs is: ",packed_input.size())
    #     packed_feats = self.models['encoder'](packed_input)
    #
    #     _, _, up_h, up_w = packed_feats[lev].size()
    #
    #     packed_feats_list = packed_feats[lev:lev + 1] \
    #                         + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in
    #                            packed_feats[lev + 1:]]
    #
    #     concated_feature = torch.cat(packed_feats_list, dim=1)
    #
    #
    #     multi_img_feat = self.models["conv2d"](torch.cat(packed_feats_list, dim=1))
    #     feats_agg = unpack_cam_feat(multi_img_feat, self.batch_size, self.num_cams)
    #
    #     vox_feat = self.models['vox_fusion'](inputs, feats_agg)
    #     dimg_deat = self.models["vox2dimg"](inputs, vox_feat)
    #
    #     img_concat =  torch.stack(dimg_deat,dim=1)
    #     B,N,C,H,W = img_concat.size()
    #
    #
    #     con_feature = []
    #     for i in range(self.fusion_level):
    #         con_feature.append(self.models["CVY_solo"][i](packed_feats[i].unsqueeze(0)))
    #     trough_cvt_feature = self.models["CVY_solo"][self.fusion_level](img_concat).view(-1,C,H,W)
    #     con_feature.append(trough_cvt_feature)
    #     outputs_depth = self.models["depth_decoder"] (con_feature)
    #
    #     # after_though = packed_feats[:lev] + [trough_cvt_feature]
    #     print("concat image shape is: ",sf_images_cur.size())
    #     print("after packed image shape is: ",packed_input.size())
    #     print("e_reault is ",type(packed_feats))
    #     print('len(pack_list)',len(packed_feats_list))
    #     print("every shape after encoder is", packed_feats_list[0].size(),  packed_feats_list[1].size(),  packed_feats_list[2].size())
    #     print("concated_feature is: ", concated_feature.size())
    #
    #     print('multi_img_feat is: ',multi_img_feat.size())
    #     print('feats_aggs is: ', feats_agg.size())
    #
    #     print("vox_feat shape is: ",vox_feat.size())
    #     print("dimg_deat shape is: ", type(dimg_deat),len(dimg_deat),dimg_deat[0].size())
    #     print("img_concat: ", img_concat.size())
    #     print("trough_cvt_feature is: ", trough_cvt_feature.size())
    #     print(con_feature[0].size(), con_feature[1].size(), con_feature[2].size())
    #
    #
    #     print("outputs keys is:", outputs_depth.keys(), outputs_depth[('disp', 0)].size())
    #     print("prev_mat is: ",inputs["prev_mat"].size())
    #     print("next_mat is: ",inputs["next_mat"].size())
    #
    #
    #     # pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}
    #     # pose_inputs = [pose_feats[-1], pose_feats[0]]
    #     # pose = self.Posenet(pose_inputs)
    #
    #     return 0