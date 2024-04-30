import torch
import argparse
import utils
# from train_model import M2depth
from models.vf_fusion import vf_fusion
import numpy as np
from PIL import Image
import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = False
# import sys

def parse_args():
    parser = argparse.ArgumentParser(description='M2Depth training script')
    # parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion.yaml', type=str,help='Config yaml file')
    parser.add_argument('--config_file', default='./configs/vf_config/ddad_surround_fusion.yaml', type=str,
                        help='Config yaml file')
    parser.add_argument('--train_model', default='vf_fusion', type=str,
                        help='abs_experment')
    parser.add_argument('--g_rank', default= 0, type=str,
                        help='single_gpu_id')
    args = parser.parse_args()
    return args

def train(cfg, args, rank):
    ab_model = args.train_model
    if ab_model == 'vf_fusion':
        print("---------------vf_fusion---------------")
        trainer = vf_fusion(cfg, rank)
        return trainer

    elif ab_model == 'sur_fusion':
        pass

    elif ab_model == 'vf_sur_fusion':
        pass

    elif ab_model == 'recons_vf_sur_fusion':
        pass

    else:
        pass

def print_data(data):
    print(type(data), data.keys())
    print(data[('K', 0)].device)
    print("idx:  ", data['idx'])
    print('sensor_name:  ', data['sensor_name'])
    print('filename:  ', data['filename'])
    print('extrinsics:  ', data['extrinsics'].size())
    print('mask: ', data['mask'].size())
    print("('K', 0) is:", data[('K', 0)].size())
    print("('inv_K', 0)", data[('inv_K', 0)].size())
    print("('color', 0, 0)", data[('color', 0, 0)].size())

    print("('color', -1, 0) ", data[('color', -1, 0)].size())

    print("('color', 1, 0) ", data[('color', 1, 0)].size())

    print("('color_aug', 0, 0)  ", data[('color_aug', 0, 0)].size())

if __name__=="__main__":

     # 9.2进行的对比试验1 vf， 今后按照这个模板来写， 注意需要进行tensorboard端口指定
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train')

    ab_model = args.train_model
    print(ab_model)

    local_rank = 2

    # local_rank = local_rank + 2
    trainer = train(cfg, args, local_rank)
    train_dataloader = trainer.dataloaders['train']
    eval_dataloader = trainer.dataloaders['eval']
    print(type(train_dataloader))
    # trainer.train()



    i = 0
    trainer.load_weights()
    trainer.debug()
    for b, data in enumerate(train_dataloader):
        # for b, data in enumerate(dataloader['eval']):
        i += 1
        print_data(data)
        # a= trainer.prossess_batch(data)
        outputs, losses = trainer.process_batch(data)

        orin = data[('color_aug', 0, 0)].detach().cpu()[0].numpy()
        depth = data['depth'].detach().cpu()[0].numpy()
        disp =  outputs[('cam', 0)][('disp', 0)].detach().cpu()[0].numpy()

        np.save('debug_files/npy/ori.npy', orin)
        np.save('debug_files/npy/orid.npy', depth)
        np.save('debug_files/npy/oridisp.npy', disp)
        # if i == 40:
        #     depth0 = outputs[('cam', 0)][('depth', 0)].detach().cpu()[0].numpy()
        #
        #     disp = outputs[('cam', 0)][('disp', 0)].detach().cpu()[0].numpy()
        #     depth1 = outputs[('cam', 1)][('depth', 0)].detach().cpu()[0].numpy()
        #     depth2 = outputs[('cam', 2)][('depth', 0)].detach().cpu()[0].numpy()
        #     depth3 = outputs[('cam', 3)][('depth', 0)].detach().cpu()[0].numpy()
        #     depth4 = outputs[('cam', 4)][('depth', 0)].detach().cpu()[0].numpy()
        #     depth5 = outputs[('cam', 5)][('depth', 0)].detach().cpu()[0].numpy()
        #     depth_ori = data["depth"].cpu().numpy()
        #
        #     np.save('debug_files/npy/depth0.npy', depth0)
        #     np.save('debug_files/npy/disp0.npy', disp)
        #     np.save('debug_files/npy/depth1.npy', depth1)
        #     np.save('debug_files/npy/depth2.npy', depth2)
        #     np.save('debug_files/npy/depth3.npy', depth3)
        #     np.save('debug_files/npy/depth4.npy', depth4)
        #     np.save('debug_files/npy/depth5.npy', depth5)
        #     np.save('debug_files/npy/depth_ori.npy', depth_ori)

        trainer.validate(b)

        print("outputs keys is: ", outputs.keys())
        print("depth is: ", data["depth"].size())
        if i >= 2:
            break
