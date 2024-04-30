import numpy as np
import torch
from collections import defaultdict
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import construct_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from models.backproject import BackprojectDepth,Project3D,SSIM
import time
from utils.logger import *
from .utils import Pose,vec_to_matrix
from .geo_cf import Projection, ViewRendering
from models.sur_depth_abs.losses.multi_cam_loss import MultiCamLoss,SingleCamLoss
import tqdm
from torchvision.utils import save_image

from models.sur_depth_abs.depth_net.cvf_depth_net import Sur_VF_DepthNet
from models.sur_depth_abs.pose_net.single_img_net.fusion_non_share.fusion_posenet import ImgFusedPoseNet
import matplotlib.pyplot as plt

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


'''
runner 构造：
初始化多卡
初始化model、dataloader、writter等一系列东西
初始化loss，变化的depth等
保存模型、tensorboard
process batch
run per epoch
train
val
'''

class Vivox_depth_ddp:
    def __init__(self, cfg, rank, st):
        # 完成模式基本参数初始化
        self.rank = rank
        self.read_config(cfg)
        self.writer = SummaryWriter(self.board_pth)
        self.log_path = os.path.join(self.log_dir, self.train_model)
        os.makedirs(os.path.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'models'), exist_ok=True)
        self.rank_start = st
        # 设置模型运算设备与通讯方式
        torch.cuda.set_device(self.rank)
        dist.init_process_group(backend='gloo')
        self.device = torch.device("cuda", self.rank)

        # 初始化model和dataloader
        self.dataloaders = {}
        self.prepare_dataset(cfg)
        self.prepare_model(cfg)
        self.prepare_middle_module(cfg)

        # 初始化optim与logger
        self.set_optimizer()
        self.load_optim()
        self.logger = Logger(cfg, True)
        self.depth_metric_names = self.logger.get_metric_names()

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


    def prepare_dataset(self, cfg):
        if self.rank == self.rank_start:
            print('### Preparing Datasets')
        # 对于任何一个rank，train的初始化都是必须的，但是只有对于一卡，才需要初始化评测集，后面的保存、显示也一样
        if self.mode == 'train':
            self.set_train_dataloader(cfg, self.rank)
            if self.rank == self.rank_start:
                self.set_eval_dataloader(cfg,self.rank)

        if self.mode == 'eval':
            self.set_eval_dataloader(cfg,self.rank)

        self.len_epoch = len(self.dataloaders['train'])

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

    def prepare_model(self,cfg):
        self.parameters_to_train = []
        self.models = {}
        self.models["depth_estimation_net"] = Sur_VF_DepthNet(cfg,self.rank)
        self.models["depth_estimation_net"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth_estimation_net"])
        self.models["depth_estimation_net"] = (self.models["depth_estimation_net"]).to(self.device)

        self.models["pose_estimation_net"] = ImgFusedPoseNet(cfg)
        self.models["pose_estimation_net"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_estimation_net"])
        self.models["pose_estimation_net"] = (self.models["pose_estimation_net"]).to(self.device)


        for v in self.models.values():
            self.parameters_to_train += list(v.parameters())

        self.load_weights()

        for key in self.models.keys():
            self.models[key] = DDP(self.models[key], device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True, broadcast_buffers=False)

    def prepare_middle_module(self,cfg):
        self.backproject_depth = {}
        self.project_3d = {}
        self.ssim = SSIM()
        # self.prepare_comparison()
        self.view_rendering, self.pose = self.init_geometry(cfg, self.rank)
        self.losses = self.init_losses(cfg, self.rank)
        self.colormap = torch.tensor([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0.5, 0, 0],
        ], dtype=torch.float32)
        self.disparity_map = plt.get_cmap('plasma', 256)

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

    def set_optimizer(self):
        self.optimizer = optim.Adam(
        self.parameters_to_train,
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            self.scheduler_step_size,
            0.1
        )

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

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)

    def to_vis_depth(self, depth_ori):

        depth_clamped = torch.clamp(depth_ori, 0.1, 50)  # 将深度值限制在 1-50m
        depth_normalized = (depth_clamped - depth_ori.min()) / (depth_ori.max() - depth_ori.min())  # 归一化到 0-1
        depth_np = depth_normalized.squeeze().cpu().numpy()
        # print(depth_np.shape)
        rgb_image = depth_np
        #rgb_image = 2.50252525 * rgb_image * rgb_image * rgb_image -2.57604895 * rgb_image * rgb_image + 1.02943279 * rgb_image - 0.03993007
        # rgb_image = cm.jet(depth_np)
        rgb_image = self.disparity_map(rgb_image)
        rgb_image = rgb_image * (1/(rgb_image.max()-rgb_image.min())) - rgb_image.min()/(rgb_image.max()-rgb_image.min())
        # rgb_image = 1 - rgb_image[:, :, :3]
        rgb_image = pow(rgb_image,0.26)
        # rgb_image = 2.50252525 * rgb_image * rgb_image * rgb_image -2.57604895 * rgb_image * rgb_image + 1.02943279 * rgb_image - 0.03993007
        rgb_image = torch.from_numpy(rgb_image)
        rgb_image = rgb_image.permute(2, 0, 1)
        return rgb_image

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"

        self.log_print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                           sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log_print(self, str):
        print(str)
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(str + '\n')

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

    def save_model(self, batch):
        """Save model weights to disk
        """
        # 应该保存的是更清晰一些的epoch+batch
        # save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.step))
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(str(self.epoch) + 'e' + '_' + str(batch) + 'b'))
        # format(str(10) + "e" + "_" + str(12000) + "b")
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
        torch.save(self.optimizer.state_dict(), save_path)

    def process_batch(self,inputs):
        # 这个函数用于在接收到单次数据后得到 loss和depth，pose等
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(self.rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(self.rank)

        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}
        # net_depth = self.models["depth_estimation_net"]
        # sur_depth = net_depth(inputs)

        # inputs[('K', 0)] = inputs[("K", 0)].float()
        # inputs[('inv_K', 0)] = inputs[("inv_K", 0)].float()
        # inputs['K', self.fusion_level + 1] = inputs['K', self.fusion_level + 1].float()
        # inputs['inv_K', self.fusion_level + 1] = inputs['inv_K', self.fusion_level + 1].float()

        net_depth = self.models["depth_estimation_net"]
        net_pose = self.models['pose_estimation_net']
        pose = self.pose.compute_pose(net_pose, inputs)
        sur_depth = net_depth(inputs)

        for cam in range(self.num_cams):
            outputs[('cam', cam)].update(pose[('cam', cam)])
            outputs[('cam', cam)].update(sur_depth[('cam', cam)])

        self.compute_depth_maps(inputs, outputs)
        # print("depth origin: ", inputs['depth'].size(), outputs['cam', 0][('depth', 0)].size())

        losses = self.compute_losses(inputs, outputs)
        # print(losses)
        return outputs, losses
        # return 0,0

    def load_optim(self):
        if self.pretrain:

            optim_file_path = os.path.join(self.load_weights_dir, 'adam.pth')
            optim_file_path = torch.load(optim_file_path,map_location=torch.device("cuda:"+str(self.rank)))
            print(optim_file_path.keys())
            self.optimizer.load_state_dict(optim_file_path)
            print("load_optim_over")

    def load_weights(self):
        load_models = self.models_to_load
        if self.pretrain:
            depth_path = os.path.join(self.load_weights_dir, 'depth_estimation_net.pth')
            pose_path = os.path.join(self.load_weights_dir, 'pose_estimation_net.pth')

            print("loading depth net weight: ",depth_path)
            depth_dict = torch.load(depth_path,map_location=torch.device("cpu"))
            pose_dict = torch.load(pose_path,map_location=torch.device("cpu"))
            # depth_dict = torch.load(depth_path,map_location=torch.device("cuda:"+str(self.rank)))
            # pose_dict = torch.load(pose_path)
            print("loading depth net weight: ", depth_path)
            self.models[load_models[0]].load_state_dict(depth_dict)
            self.models[load_models[1]].load_state_dict(pose_dict)
            print("load_model_over")
            # print(optim_file_path)


    def tensor_board_writer(self,inputs,outputs,losses,batch_idx):
        if batch_idx * (self.epoch + 1) % 10 == 0:
            if batch_idx * (self.epoch + 1) % 50 == 0:
                self.writer.add_image("image/depth_image_0",
                                      self.to_vis_depth(outputs[('cam', 1)][('depth', 0)].detach().cpu()[0]),
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/depth_image_1",
                                      self.to_vis_depth(outputs[('cam', 0)][('depth', 0)].detach().cpu()[0]),
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/depth_image_2",
                                      self.to_vis_depth(outputs[('cam', 2)][('depth', 0)].detach().cpu()[0]),
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/depth_image_3",
                                      self.to_vis_depth(outputs[('cam', 4)][('depth', 0)].detach().cpu()[0]),
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/depth_image_4",
                                      self.to_vis_depth(outputs[('cam', 5)][('depth', 0)].detach().cpu()[0]),
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/depth_image_5",
                                      self.to_vis_depth(outputs[('cam', 3)][('depth', 0)].detach().cpu()[0]),
                                      (self.len_epoch * self.epoch + batch_idx) // 50)

                self.writer.add_image("image/origin_image_0", inputs[('color', 0, 0)].detach().cpu()[0][1],
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/origin_image_1", inputs[('color', 0, 0)].detach().cpu()[0][0],
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/origin_image_2", inputs[('color', 0, 0)].detach().cpu()[0][2],
                                      (self.len_epoch * self.epoch + batch_idx)  // 50)
                self.writer.add_image("image/origin_image_3", inputs[('color', 0, 0)].detach().cpu()[0][4],
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/origin_image_4", inputs[('color', 0, 0)].detach().cpu()[0][5],
                                      (self.len_epoch * self.epoch + batch_idx) // 50)
                self.writer.add_image("image/origin_image_5", inputs[('color', 0, 0)].detach().cpu()[0][3],
                                      (self.len_epoch * self.epoch + batch_idx)  // 50)

                print("save board: ", (self.len_epoch * self.epoch + batch_idx) )
            self.writer.add_scalar('loss/reproj_loss', losses['reproj_loss'], 1+(self.len_epoch * self.epoch + batch_idx) // 10)
            self.writer.add_scalar('loss/spatio_loss', losses['spatio_loss'], 1+(self.len_epoch * self.epoch + batch_idx) // 10)
            self.writer.add_scalar('loss/spatio_tempo_loss', losses['spatio_tempo_loss'], 1+(self.len_epoch * self.epoch + batch_idx) // 10)
            self.writer.add_scalar('loss/smooth', losses['smooth'], 1+(self.len_epoch * self.epoch + batch_idx) // 10)
            self.writer.add_scalar('total_loss', losses['total_loss'],1+ (self.len_epoch * self.epoch + batch_idx) // 10)
        else:
            pass

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
        # print(depth_eval_metric)
        # print(depth_eval_median)
        self.writer.add_scalar('val/metric', depth_eval_metric['abs_rel'], (self.len_epoch * self.epoch + b)  // 200)
        self.writer.add_scalar('val/median', depth_eval_median['abs_rel'], (self.len_epoch * self.epoch + b)  // 200)
        self.logger.log_tb('val', inputs, outputs, losses, self.step)
        del inputs, outputs, losses

        self.set_train()

    def run_per_epoch(self):
        torch.autograd.set_detect_anomaly(True)

        print("----------------------------Start Training--------------------------------------")
        self.set_train()

        for batch_idx, inputs in enumerate(self.dataloaders['train']):
            # print("training-epoch")

            before_op_time = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            outputs, losses = self.process_batch(inputs)
            losses["total_loss"].backward()
            self.optimizer.step()

            if self.rank == self.rank_start:
                self.tensor_board_writer(inputs, outputs, losses, batch_idx)

            duration = time.time() - before_op_time
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.early_log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                if self.rank == self.rank_start:
                    self.log_time(batch_idx, duration, losses["total_loss"].cpu().data)
                print("losses: ", losses)
                # print("mid_ser:",batch_idx, duration, losses["total_loss"].cpu().data)

            if self.step % self.log_frequency == 0  and self.log_frequency > 0 and self.rank == self.rank_start:
                print("save_model: ",self.step)
                self.save_model(batch_idx)
            if self.step % 400 == 0 and self.rank == self.rank_start:
                print("start_val-----")
                self.validate(batch_idx)
            self.step += 1
        self.lr_scheduler.step()

    def train(self):
        self.epoch = 0
        self.start_time = time.time()
        self.step = 1
        print("---------------------------------length of per epoch is: {}".format(self.len_epoch))
        if self.rank == self.rank_start:
            val_dataloader = self.dataloaders['eval']
            self.val_iter = iter(val_dataloader)

        for self.epoch in range(self.num_epochs):
            print("----------epoch------------", "rank is" + str(self.rank), self.epoch)
            if self.ddp_enable:
                self.train_sampler.set_epoch(self.epoch)
            # self.dataloaders['train'].sampler.set_epoch(epoch)
            self.run_per_epoch()
            if self.ddp_enable:
                dist.barrier()
        print("over training_process: ",self.epoch)
        print("-"*100)

    def save_depth(self,outputs_data, inputs,b):

        np.save('./debug_files/eval_files/d0.npy',outputs_data['cam', 0][('depth', 0)].cpu().numpy() )
        np.save('./debug_files/eval_files/d1.npy',outputs_data['cam', 1][('depth', 0)].cpu().numpy() )
        np.save('./debug_files/eval_files/d2.npy',outputs_data['cam', 2][('depth', 0)].cpu().numpy() )
        np.save('./debug_files/eval_files/d3.npy',outputs_data['cam', 3][('depth', 0)].cpu().numpy() )
        np.save('./debug_files/eval_files/d4.npy',outputs_data['cam', 4][('depth', 0)].cpu().numpy() )
        np.save('./debug_files/eval_files/d5.npy',outputs_data['cam', 5][('depth', 0)].cpu().numpy() )

    def save_mask(self,mask):
        save_image(mask[:, 0, ...], './debug_files/eval_files/0.png')
        save_image(mask[:, 1, ...], './debug_files/eval_files/1.png')
        save_image(mask[:, 2, ...], './debug_files/eval_files/2.png')
        save_image(mask[:, 3, ...], './debug_files/eval_files/3.png')
        save_image(mask[:, 4, ...], './debug_files/eval_files/4.png')
        save_image(mask[:, 5, ...], './debug_files/eval_files/5.png')

    def write_dis_txt(self,fname, d_list):
        with open(fname, 'w') as f:
            with open(fname, 'w') as f:
                f.writelines("{}\n".format(str(item))  for item in d_list)

    def evaluate(self):
        print("final2")
        av_per_eve = [(0,0)]



        print("--------------------------------------start eval----------------------------")
        eval_dataloader = self.dataloaders['eval']
        train_dataloader = self.dataloaders['train']
        print("dataloader have: ", len(train_dataloader),len(eval_dataloader))


        # print("num eval data: ",len(eval_dataloader))
        # 我们模型的参数在初始化的时候就已经被设置了
        self.set_eval()
        # process = tqdm(eval_dataloader)

        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)
        d_list = []

        i = 0
        for batch_idx, inputs in enumerate(eval_dataloader):
            # print("inputs: ", inputs.keys())
            if i > 0:
                break
            i+=1
            # print(inputs['mask'].size())
            # self.save_mask(inputs['mask'])
            # visualize synthesized depth maps
            # print(inputs.keys())
            # print("idx:", inputs['idx'])
            # print('dataset_idx', inputs['dataset_idx'])
            # print( 'filename', inputs[ 'filename'])

            with torch.no_grad():

                outputs, _ = self.process_batch(inputs)
                # self.save_depth(outputs,inputs)
                depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs)

                if depth_eval_median['abs_rel'] > 0.26:
                    d_list.append((inputs['filename'], depth_eval_metric['abs_rel'],depth_eval_median['abs_rel']))

                # print("depth_eval_metric: ", depth_eval_metric)
                # print("depth_eval_median: ", depth_eval_median)
                for key in self.depth_metric_names:
                    avg_depth_eval_metric[key] += depth_eval_metric[key]
                    avg_depth_eval_median[key] += depth_eval_median[key]
                # if batch_idx % 20 ==0:\
                #     print("------------------",batch_idx,"-----------------")
                #     print("depth_eval_metric: ",depth_eval_metric)
                #     print("depth_eval_median: ",depth_eval_median)

                if batch_idx % 80 == 0:
                    i = batch_idx // 80
                    m_metric, m_median =  avg_depth_eval_metric['abs_rel'],avg_depth_eval_median['abs_rel']
                    if batch_idx != 0:
                        print("now final is: ", m_metric/(batch_idx+1), m_median/(batch_idx+1))
                        print("local index is {} ---".format(i), (m_metric-av_per_eve[i][0])/80, (m_median-av_per_eve[i][1])/80)
                    av_per_eve.append((m_metric,m_median))
                    print("d_list has: ",len(d_list))
                if batch_idx % 100 == 0:
                    print("--------------------b: ",batch_idx)

            torch.cuda.empty_cache()

        for key in self.depth_metric_names:
            avg_depth_eval_metric[key] /= len(eval_dataloader)
            avg_depth_eval_median[key] /= len(eval_dataloader)

        print('Evaluation result...\n')
        self.logger.print_perf(avg_depth_eval_metric, 'metric')
        self.logger.print_perf(avg_depth_eval_median, 'median')

        print("metric: ",avg_depth_eval_metric)
        print("median ", avg_depth_eval_median )
        self.write_dis_txt('dis_199.txt',d_list)

