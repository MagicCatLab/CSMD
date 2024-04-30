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
from torch.utils.tensorboard import SummaryWriter


_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

class Base_runner:
    def __init__(self, cfg, rank):
        self.rank = rank

    def read_config(self, cfg):
        raise NotImplementedError('Not implemented for BaseModel')

    def prepare_dataset(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def set_optimizer(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def load_weights(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def set_train(self):
        self.mode = 'train'
        for m in self.models.values():
            m.train()

    def set_val(self):
        self.mode = 'val'
        for m in self.models.values():
            m.eval()

    def prepare_model(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def set_train_dataloader(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def set_eval_dataloader(self):
        raise NotImplementedError('Not implemented for BaseModel')



