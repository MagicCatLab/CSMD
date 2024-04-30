import torch
import argparse
import utils
# from train_model import M2depth
from models.vox_lom_depth_ddp import Vox_lom_depth_ddp

# 现阶段对比实验二的内容

def parse_args():
    parser = argparse.ArgumentParser(description='M2Depth training script')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion.yaml', type=str,help='Config yaml file')
    # parser.add_argument('--config_file', default='./configs/sur_vf_ddp/ddad_surround_fusion_ddp.yaml', type=str,
    #                     help='Config yaml file')
    # parser.add_argument('--config_file', default='./configs/sur_vf_ddp/nusc_surround_fusion_ddp.yaml', type=str,
    #                     help='Config yaml file')

    parser.add_argument('--config_file', default='./configs/lomvox_ddp/ddad_surround_fusion_ddp.yaml', type=str,
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
        trainer = Vox_lom_depth_ddp(cfg, rank)
        return trainer

    elif ab_model == 'sur_fusion':
        pass

    elif ab_model == 'vf_sur_fusion':
        pass

    elif ab_model == 'recons_vf_sur_fusion':
        pass

    else:
        pass


if __name__=="__main__":

     # 9.2进行的对比试验1 vf， 今后按照这个模板来写， 注意需要进行tensorboard端口指定
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train')

    ab_model = args.train_model
    print(ab_model)
    local_rank = args.local_rank
    local_rank = local_rank + 0
    print(local_rank)
    # local_rank = local_rank + 2
    trainer = train(cfg, args, local_rank)
    # train_dataloader = trainer.dataloaders['train']
    # print(len(train_dataloader))
    #
    # i=0
    # for batch_idx, inputs in enumerate(train_dataloader):
    #      i+=1
    #      outputs,losses = trainer.process_batch(inputs)
    #      print("outputs keys is: ", outputs.keys())
    #      print(losses)
    #      if i>0:
    #          break


    # # eval_dataloader = trainer.dataloaders['eval']
    # print(type(train_dataloader))
    trainer.train()

    # trainer.evaluate()

