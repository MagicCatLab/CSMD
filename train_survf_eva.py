import torch
import argparse
import utils
# from train_model import M2depth
from models.sur_depth_eva import Sur_depth_ddp_eva

gpu_i = 1

def parse_args():
    parser = argparse.ArgumentParser(description='M2Depth training script')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')
    # # parser.add_argument('--config_file', default='./configs/ddad/ddad_surround_fusion.yaml', type=str,help='Config yaml file')
    parser.add_argument('--config_file', default='./configs/sur_vf_ddp/ddad_surround_fusion_eva.yaml', type=str,
                        help='Config yaml file')


    # parser.add_argument('--config_file', default='./configs/sur_vf_ddp/nusc_surround_fusion_eva.yaml', type=str,
    #                     help='Config yaml file')
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
        trainer = Sur_depth_ddp_eva(cfg, rank,gpu_i)
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
    local_rank = local_rank + gpu_i
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
    # trainer.train()

    # trainer.vis_rel_pixbdep()

    print("train_eval length: ",len(trainer.dataloaders['train']),len(trainer.dataloaders['eval']))
    # s = './data/DDAD/depth_save/surf_depth_ab2_weights_5e_994b/scene_150_200'
    # s = './data/DDAD/depth_save/nusc/scene_150_200'
    # s = './data/DDAD/depth_save/nusc_vf/scene_150_200'
    # 是否储存差值 与是否储存预测图像
    # s = './data/disk3/depth_all_save/ddad_vfm_2/scene_150_200'
    # s = './data/disk3/depth_all_save/ddad_myself/scene_150_200'
    scene = 2733
     # 这个是全有
     #  load_weights_dir: './results_ab/nusc/sur_vf_fusion_t3_gt/survf_fusion_nusc/models/weights_0e_2399b'
    # s = '/home/ps/disk3/depth_all_save/nusc_compare_my_id_only/'

     # 对比试验，这个是vfdepth，也是对比实验中的全无
     #  load_weights_dir: './results_ab/nusc/sur_vf_fusion_final/survf_fusion_nusc/models/weights_0e_249b'
     #   metric | abs_rel: 0.318 | sq_rel: 7.029 | rms: 8.276 | log_rms: 0.349 | a1: 0.703 | a2: 0.868 | a3: 0.929
     #   median | abs_rel: 0.268 | sq_rel: 4.675 | rms: 7.965 | log_rms: 0.334 | a1: 0.711 | a2: 0.875 | a3: 0.933
    #s = '/home/ps/disk3/depth_all_save/nusc_compare_ab_c_id_only/'

     # # 对比试验，这个是缺少loss的引入
     # load_weights_dir: './results_ab/nusc/sur_vf_fusion_final/survf_fusion_nusc/models/weights_0e_199b'
    #s = '/home/ps/disk3/depth_all_save/nusc_compare_ab_l_id_only/'

     # 对比试验，这个是缺少overlap的处理
     # load_weights_dir: './results_ab/nusc/sur_vf_fusion_t3_gt/survf_fusion_nusc/models/weights_0e_1499b'
    s = '/home/ps/disk3/depth_all_save/ddad_compare_ab_o_id/sc29/'
    trainer.evaluate(save_root=s, save_del=False,save_pred=False)
    print(s)
    # trainer.evaluate(save_root='./data/nuscenes/depth_save/surf_depth')



    print("train_eval length: ", len(trainer.dataloaders['train']), len(trainer.dataloaders['eval']))
    # # print("no loss")
    # print("no overlap")
    print("all_got")
    # 2023 09 22 评测就用这个
