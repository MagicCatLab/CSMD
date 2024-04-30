import torch
import argparse
import utils
# from train_model import M2depth
from models.sur_m2depth_abs_ddp import Sur_M2depth_ddp
import numpy as np
from PIL import Image
import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

def parse_args():
    parser = argparse.ArgumentParser(description='M2Depth training script')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion.yaml', type=str,help='Config yaml file')
    parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion_or.yaml', type=str,
                        help='Config yaml file')
    args = parser.parse_args()
    return args