import torch
import argparse
import utils
# from train_model import M2depth
# from models.M2depth import M2depth
from models.sur_m2depth_abs_ddp import Sur_M2depth_ddp
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
# import sys
# sys.path.append('~/lrh_root/Paper1/M2depth/external/dgp/dgp')

def set_train_dataloader(cfg,rank):
    height = 384
    width = 640
    _augmentation = {
        'image_shape': (int(height), int(width)),
        'jittering': (0.2, 0.2, 0.2, 0.05),
        'crop_train_borders': (),
        'crop_eval_borders': ()
    }

def parse_args():
    parser = argparse.ArgumentParser(description='M2Depth training script')
    # parser.add_argument('--config_file', default ='./configs/DDAD/surround.yaml', type=str, help='Config yaml file')
    # parser.add_argument('--config_file', default ='./configs/nuscenes/nusc_surround_fusion.yaml', type=str, help='Config yaml file')
    parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion.yaml', type=str,help='Config yaml file')
    args = parser.parse_args()
    return args

def train(cfg):
    writer = SummaryWriter('./path/to/log')
    trainer = Sur_M2depth_ddp(cfg, 0, writer)

    return trainer

def save_figs(img_file,file_name,dataset_name):

    # 遍历每个图像并保存
    for i in range(img_file.shape[1]):
        # image = Image.fromarray(img_file[0, i])

        img_array = img_file[0, i].numpy()  # 将Tensor转换为NumPy数组
        # print(img_array.shape)
        img_array = (img_array * 255).astype(np.uint8)

        image = Image.fromarray(img_array.transpose(1, 2, 0))
        image.save("debug_files/"+dataset_name+"/image_"+file_name+f"{i}.png")


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

    # save_figs(data[('color', 0, 0)],'color', 'ddad')
    # save_figs(data[('color', -1, 0)],'color-1', 'ddad')
    # save_figs(data[('color', 1, 0)],'color1', 'ddad')
    # save_figs(data[('color_aug', 0, 0)],'color_aug', 'ddad')

    # save_figs(data[('color', 0, 0)],'color', 'nuscenes')
    # save_figs(data[('color', -1, 0)],'color-1', 'nuscenes')
    # save_figs(data[('color', 1, 0)],'color1', 'nuscenes')
    # save_figs(data[('color_aug', 0, 0)],'color_aug', 'nuscenes')

def save_dfigs(file_path, outputs, inputs, i ):
    depth_origin = inputs["depth"].cpu().numpy()
    image_origin = inputs[('color_aug', 0, 0)].cpu().numpy()
    outputs_depth = outputs[('disp', 0)].detach().cpu().numpy()
    np.save(file_path+"____"+str(i)+"____"+'depth_origin.npy', depth_origin)
    np.save(file_path+"____"+str(i)+"____"+'image_origin.npy', image_origin)
    np.save(file_path+"____"+str(i)+"____"+'outputs_origin.npy', outputs_depth)


if __name__=="__main__":
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train')
    # cfg = utils.get_config(args.config_file, mode='eval')
    print("training type is: ",cfg["datatype"]["dataset"])

    if cfg['ddp']['ddp_enable'] == True:
        print("multi")
        trainer = train(cfg)
        dataloader = trainer.dataloaders
        print(type(dataloader))
        print(type(dataloader))
        print(dataloader.keys())
        print('Number of batches:',len(dataloader['train']))
        # print('Number of batches:',len(dataloader['eval']))
        i = 0
        # trainer.train()
        for b, data in enumerate(dataloader['train']):
        # for b, data in enumerate(dataloader['eval']):
            i += 1
            print_data(data)
            # a= trainer.prossess_batch(data)
            outputs, losses = trainer.process_batch(data)
            print("depth is: ", data["depth"].size())
            print("losses: ", losses)
            print("ouputs depth is: ", outputs[('cam', 0)].keys(), outputs[('cam', 0)][('depth', 0)].size())
            print("outputs keys is: ", outputs.keys())
            print("original image is: ", data[('color', 0, 0)].size())

            # save_dfigs("./midlle_files/ddad/",outputs,data,i)
            if i>=1:
                break

        # j=0
        # for b, data in enumerate(dataloader['eval']):
        # # for b, data in enumerate(dataloader['eval']):
        #     j += 1
        #     print("eval data keys: ", data.keys())
        #
        #     print("eval image is: ",data[('color_aug', 0, 0)].size())
        #     if j>=1:
        #         break

    else:
        trainer = train(cfg)
        dataloader = trainer.dataloaders
        print(type(dataloader))
        print(type(dataloader))
        print(dataloader.keys())
        # print('Number of batches:',len(dataloader['train']))
        # print('Number of batches:',len(dataloader['eval']))
        i = 0

        for b, data in enumerate(dataloader['train']):
        # for b, data in enumerate(dataloader['eval']):
            i += 1
            print_data(data)
            # a= trainer.prossess_batch(data)
            a = trainer.process_batch(data)
            if i>=1:
                break