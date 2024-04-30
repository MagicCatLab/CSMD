import torch
import argparse
import utils
# from train_model import M2depth
from models.M2depth_ddp import M2depth_ddp
import numpy as np
from PIL import Image
import os
import torch.distributed as dist
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
# import sys
# sys.path.append('~/lrh_root/Paper1/M2depth/external/dgp/dgp')


def parse_args():
    parser = argparse.ArgumentParser(description='M2Depth training script')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--config_file', default ='./configs/DDAD/surround.yaml', type=str, help='Config yaml file')
    parser.add_argument('--config_file', default ='./configs/nuscenes/nusc_surround_fusion.yaml', type=str, help='Config yaml file')
    # parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion.yaml', type=str,help='Config yaml file')

    args = parser.parse_args()
    return args

def train(cfg,rank):

    trainer = M2depth_ddp(cfg, rank)

    return trainer

def ddp_setup(rank, world_size):
    """
    Args:
        rank: 进程的唯一标识，在 init_process_group 中用于指定当前进程标识
        world_size: 进程总数
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


if __name__=="__main__":
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train')
    # cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    # cfg = utils.get_config(args.config_file, mode='eval')
    print("training type is: ",cfg["datatype"]["dataset"])
    local_rank = args.local_rank
    print("now local_rank is: ", local_rank)
    local_rank = local_rank +1
    if cfg['ddp']['ddp_enable'] == True:
        print("multi")
        trainer = train(cfg,local_rank)
        dataloader = trainer.dataloaders
        i = 0
        trainer.train()
