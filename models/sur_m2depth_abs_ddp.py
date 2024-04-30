import torch
from collections import defaultdict
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import construct_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from models.model_component.norm_type import *
from models.model_component.resnet_encoder import ResnetEncoder
from models.model_component.PoseEstimation.Posenet import IF_PoseNet

from models.model_component.depth_estimation_net import Depth_EstimationNet
from models.backproject import BackprojectDepth,Project3D,SSIM
import time
from utils.logger import *
from models.vf_module.depth_net.fusion_depthnet import FusedDepthNet
from models.vf_module.pose_net.fusion_posenet import FusedPoseNet
from .utils import Pose,vec_to_matrix
from .geo_cf import Projection, ViewRendering
from models.vf_module.losses.multi_cam_loss import MultiCamLoss,SingleCamLoss
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


class Sur_M2depth_ddp:
    def __init__(self, cfg, rank, writer):
        self.rank = rank
        self.writer = writer
        # super(M2depth, self).__init__(cfg)
        self.read_config(cfg)
        print("rank is"+str(self.rank)+"    read_conig over")
        self.log_path = os.path.join(self.log_dir, self.model_name)
        os.makedirs(os.path.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'models'), exist_ok=True)

        self.dataloaders = {}
        self.colormap = torch.tensor([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0.5, 0, 0],
        ], dtype=torch.float32)

        # rank, world_size = get_dist_info()
        # self.rank = rank

        torch.cuda.set_device(self.rank)
        print("rank is"+str(self.rank)+"    set_device over")
        dist.init_process_group(backend='nccl')
        print("rank is"+str(self.rank)+"    nccl over")


        # dist.init_process_group(backend='nccl')
        self.device = torch.device("cuda", self.rank)

        self.ssim = SSIM()
        print("rank is"+str(self.rank)+"    ssim over")
        self.prepare_dataset(cfg, rank)
        print("rank is"+str(self.rank)+"    dataset oevr")
        self.train_dataloader = self.dataloaders['train']
        self.parameters_to_train = []
        # self.models = self.prepare_model(cfg,rank)
        # self.prepare_model(cfg,rank)
        self.prepare_cf_model(cfg,rank)
        print("rank is"+str(self.rank)+"    model over")
        self.backproject_depth = {}
        self.project_3d = {}
        # self.prepare_comparison()
        self.view_rendering, self.pose = self.init_geometry(cfg, rank)
        self.losses = self.init_losses(cfg, rank)
        # self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer,
            self.scheduler_step_size,
            0.1
        )


        self.logger = Logger(cfg, True)
        self.depth_metric_names = self.logger.get_metric_names()


    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose

    def init_losses(self, cfg, rank):
        if self.aug_depth:
            # loss_model = DepthSynLoss(cfg, rank)
            pass
        elif self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else :
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model

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
    def prepare_cf_model(self,cfg,rank):
        self.models = {}
        self.models["depth_estimatiion_net"] = FusedDepthNet(cfg)
        self.models["depth_estimatiion_net"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth_estimatiion_net"])
        self.models["depth_estimatiion_net"] = (self.models["depth_estimatiion_net"]).to(self.device)

        self.models["pose_estimation_net"] = FusedPoseNet(cfg)
        self.models["pose_estimation_net"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_estimation_net"])
        self.models["pose_estimation_net"] = (self.models["pose_estimation_net"]).to(self.device)


        for v in self.models.values():
            self.parameters_to_train += list(v.parameters())
        for key in self.models.keys():
            self.models[key] = DDP(self.models[key], device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True, broadcast_buffers=False)

        # for key in self.models.keys():
        #     self.models[key] = DDP(self.models[key], device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True, broadcast_buffers=False)


    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')

        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            # self.set_eval_dataloader(cfg, rank)

            if rank == self.rank:
                self.set_eval_dataloader(cfg,rank)

        if self.mode == 'eval':
            self.set_eval_dataloader(cfg,rank)


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



    def predict_pose(self, inputs):
        """
        This function predicts poses.
        """
        net = None
        if (self.mode != 'train') and self.ddp_enable:
            net = self.models['pose_net'].module
        else:
            net = self.models['pose_net']

        pose = self.pose.compute_pose(net, inputs)
        return pose

    def compute_depth_maps(self, inputs, outputs):
        """
        This function computes depth map for each viewpoint.
        """
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', scale)] = self.to_depth(disp, ref_K)
                if self.aug_depth:
                    disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                    outputs[('cam', cam)][('depth', scale, 'aug')] = self.to_depth(disp, ref_K)

    def process_batch(self,inputs):
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(self.rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(self.rank)
        # for key, ipt in inputs.items():
        #     if key not in _NO_DEVICE_KEYS:
        #         print(inputs[key].device)

        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        inputs[('K', 0)] = inputs[("K",0)].float()
        inputs[('inv_K', 0)] = inputs[("inv_K",0)].float()
        inputs['K',self.fusion_level+1] = inputs['K',self.fusion_level+1].float()
        inputs['inv_K',self.fusion_level+1] = inputs['inv_K',self.fusion_level+1].float()

        net_depth = None
        # if (self.mode != 'train') and self.ddp_enable:
        #     net_depth = self.models["depth_estimatiion_net"].module
        # else:
        net_depth = self.models["depth_estimatiion_net"]
        sur_depth = net_depth(inputs)
        # print("depth_feats is: ", sur_depth.keys())

        net_pose = None
        # if (self.mode != 'train') and self.ddp_enable:
        #     net_pose = self.models['pose_estimation_net'].module
        # else:
        net_pose = self.models['pose_estimation_net']
        pose = self.pose.compute_pose(net_pose, inputs)
        # print("pose is: ", pose.keys())
        for cam in range(self.num_cams):
            outputs[('cam', cam)].update(pose[('cam', cam)])
            outputs[('cam', cam)].update(sur_depth[('cam', cam)])

        self.compute_depth_maps(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs,losses

    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        # generate image and compute loss per cameara
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)

            losses += cam_loss
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams

        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))

        loss_mean['total_loss'] = losses
        return loss_mean

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)

    def to_depth(self, disp_in, K_in):
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale

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




    def validate(self,b):
        """
        This function validates models on validation dataset to monitor training process.
        """
        self.set_eval()
        inputs = next(self.val_iter)

        outputs, losses = self.process_batch(inputs)

        # if 'depth' in inputs:
        depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, vis_scale=True)
        self.logger.print_perf(depth_eval_metric, 'metric')
        self.logger.print_perf(depth_eval_median, 'median')
        print(depth_eval_metric)
        print(depth_eval_median)
        self.writer.add_scalar('val/metric', depth_eval_metric['abs_rel'], b * (self.epoch + 1) / 10)
        self.writer.add_scalar('val/median', depth_eval_median['abs_rel'], b * (self.epoch + 1) / 10)
        self.logger.log_tb('val', inputs, outputs, losses, self.step)
        del inputs, outputs, losses

        self.set_train()

    def to_vis_depth(self, depth_ori):
        # depth_clamped = torch.clamp(depth_ori, 1, 50)  # 将深度值限制在 1-50m
        # depth_normalized = (depth_clamped - 1) / (50 - 1)  # 归一化到 0-1
        # # depth_normalized = (depth_ori - depth_ori.min()) / (depth_ori.max() - depth_ori.min())
        #
        # # 转换为灰度图像（在这里我们只需要重复相同的值到 RGB 三个通道）
        # gray_image = depth_normalized.repeat(3, 1, 1)
        # return gray_image

        depth_clamped = torch.clamp(depth_ori, 0.1, 50)  # 将深度值限制在 1-50m
        depth_normalized = (depth_clamped - depth_ori.min()) / (depth_ori.max() - depth_ori.min())  # 归一化到 0-1

        depth_np = depth_normalized.squeeze().cpu().numpy()

        # 使用 matplotlib 的 colormap 将深度图转换为 RGB 图像
        rgb_image = cm.jet(depth_np)
        rgb_image = rgb_image[:, :, :3]
        rgb_image = torch.from_numpy(rgb_image)
        rgb_image = rgb_image.permute(2,0,1)
        # print("rgb_image is: ",rgb_image.size())
        return rgb_image


    def run_per_epoch(self,epoch):
        torch.autograd.set_detect_anomaly(True)
        if self.rank == 0:
            print("Start Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.dataloaders['train']):
            # print("training-epoch")

            before_op_time = time.time()
            self.model_optimizer.zero_grad()
            outputs, losses = self.process_batch(inputs)
            losses["total_loss"].backward()

            if batch_idx * (epoch+1) % 10 == 0:
                if batch_idx * (epoch + 1) % 50 == 0:
                    self.writer.add_image("image/depth_image_0",
                                          self.to_vis_depth(outputs[('cam', 0)][('depth', 0)].detach().cpu()[0]),
                                          batch_idx * (epoch + 1) / 50)
                    self.writer.add_image("image/depth_image_1",
                                          self.to_vis_depth(outputs[('cam', 1)][('depth', 0)].detach().cpu()[0]),
                                          batch_idx * (epoch + 1) / 50)
                    self.writer.add_image("image/depth_image_2",
                                          self.to_vis_depth(outputs[('cam', 2)][('depth', 0)].detach().cpu()[0]),
                                          batch_idx * (epoch + 1) / 50)
                    self.writer.add_image("image/origin_image_0", inputs[('color', 0, 0)].detach().cpu()[0][0],
                                          batch_idx * (epoch + 1) / 50)
                    self.writer.add_image("image/origin_image_1", inputs[('color', 0, 0)].detach().cpu()[0][1],
                                          batch_idx * (epoch + 1) / 50)
                    self.writer.add_image("image/origin_image_2", inputs[('color', 0, 0)].detach().cpu()[0][2],
                                          batch_idx * (epoch + 1) / 50)
                    print("save board: ",batch_idx * (epoch+1))
                self.writer.add_scalar('loss/reproj_loss', losses['reproj_loss'], batch_idx * (epoch+1)/10)
                self.writer.add_scalar('loss/spatio_loss', losses['spatio_loss'],batch_idx * (epoch+1)/10)
                self.writer.add_scalar('loss/spatio_tempo_loss', losses['spatio_tempo_loss'], batch_idx * (epoch+1)/10)
                self.writer.add_scalar('loss/smooth', losses['smooth'], batch_idx * (epoch+1)/10)
                self.writer.add_scalar('total_loss', losses['total_loss'], batch_idx * (epoch+1)/10)
                # print("save board: ", batch_idx * (epoch + 1) / 10)

                # print("depth_size: ", outputs[('cam', 0)][('depth', 0)].size())


            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["total_loss"].cpu().data)
                print("mid_ser:",batch_idx, duration, losses["total_loss"].cpu().data)
                # print(losses)

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

            if self.step % self.log_frequency == 0  and self.log_frequency > 0 :
                self.save_model()
            #


            if self.step % 50 == 0  and self.rank == 0:
                print("start_val-----")
                self.validate(batch_idx)
            # if self.logger.is_checkpoint(self.step)  and self.rank == 0:
            #     print("start_val-----")
            #     self.validate(batch_idx)

            self.step += 1
        self.lr_scheduler.step()
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

        if self.pretrain:
            for n in self.models_to_load:
                path = os.path.join(self.load_weights_dir, 'depth_net.pth')
                print("path------:",path)
                pre_trained_dict = torch.load(path)
                self.models[n].load_state_dict(pre_trained_dict)
                print("load_over")

        self.epoch = 0
        self.start_time = time.time()
        self.step = 1

        if self.rank == 0:
            val_dataloader = self.dataloaders['eval']
            self.val_iter = iter(val_dataloader)

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print("----------epoch------------","rank is"+str(self.rank),epoch)
            self.dataloaders['train'].sampler.set_epoch(epoch)
            self.run_per_epoch(epoch)
            dist.barrier()

    def eval(self):
        if self.pretrain:
            for n in self.models_to_load:
                path = os.path.join(self.load_weights_dir, 'depth_net.pth')
                print("path------:",path)
                pre_trained_dict = torch.load(path)
                self.models[n].load_state_dict(pre_trained_dict)
                print("load_over")
        self.set_eval()
        eval_dataloader = self.dataloaders['eval']
        # process = tqdm(eval_dataloader)

        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)

        for batch_idx, inputs in enumerate(eval_dataloader):
            outputs, _ = self.process_batch(inputs)
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs)
            if batch_idx%10 ==0:
                print(depth_eval_metric)
                print(depth_eval_median)

            for key in self.depth_metric_names:
                avg_depth_eval_metric[key] += depth_eval_metric[key]
                avg_depth_eval_median[key] += depth_eval_median[key]
        for key in self.depth_metric_names:
            avg_depth_eval_metric[key] /= len(eval_dataloader)
            avg_depth_eval_median[key] /= len(eval_dataloader)

        print("final:",avg_depth_eval_median)
        print("final:",avg_depth_eval_metric)
