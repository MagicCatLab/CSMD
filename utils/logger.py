# Copyright (c) 2023 42dot. All rights reserved.
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as pil

from tensorboardX import SummaryWriter

from .visualize import colormap
from .misc import pretty_ts, cal_depth_error, cal_depth_error_del


def set_tb_title(*args):
    """
    This function sets title for tensorboard plot.
    """    
    title = ''
    for i, s in enumerate(args):
        if not i%2: title += '/'
        s = s if isinstance(s, str) else str(s)
        title += s
    return title[1:]

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def resize_for_tb(image):
    """
    This function resizes images for tensorboard plot.
    """     
    h, w = image.size()[-2:]
    return F.interpolate(image, [h//2, w//2], mode='bilinear', align_corners=True) 
    

def plot_tb(writer, step, img, title, j=0):
    """
    This function plots images on tensotboard.
    """     
    img_resized = resize_for_tb(img)    
    writer.add_image(title, img_resized[j].data, step)


def plot_norm_tb(writer, step, img, title, j=0):
    """
    This function plots normalized images on tensotboard.
    """     
    img_resized = torch.clamp(resize_for_tb(img), 0., 1.)
    writer.add_image(title, img_resized[j].data, step)


def plot_disp_tb(writer, step, disp, title, j=0):
    """
    This function plots disparity maps on tensotboard.
    """  
    disp_resized = resize_for_tb(disp).float()
    disp_resized = colormap(disp_resized[j, 0])
    writer.add_image(title, disp_resized, step)    

    
class Logger:
    """
    Logger class to monitor training
    """
    def __init__(self, cfg, use_tb=True):
        self.read_config(cfg)
        os.makedirs(self.log_path, exist_ok=True)
        
        if use_tb: 
            self.init_tb()
            
        if self.eval_visualize:
            self.init_vis()

        self._metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
        
    def init_tb(self):
        self.writers = {}
        for mode in ['train', 'val']:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
    def close_tb(self):
        for mode in ['train', 'val']:
            self.writers[mode].close()

    def init_vis(self):
        vis_path = os.path.join(self.log_path, 'vis_results')
        os.makedirs(vis_path, exist_ok=True)
        
        self.cam_paths = []
        for cam_id in range(self.num_cams):
            cam_path = os.path.join(vis_path, f'cam{cam_id:d}')
            os.makedirs(cam_path, exist_ok=True)
            self.cam_paths.append(cam_path)
            
        if self.syn_visualize:
            self.syn_path = os.path.join(self.log_path, 'syn_results')
            os.makedirs(self.syn_path, exist_ok=True)
            
    def get_metric_names(self):
        return self._metric_names
    
    def update(self, mode, epoch, world_size, batch_idx, step, start_time, before_op_time, inputs, outputs, losses):
        """
        Display logs with respect to the log frequency
        """    
        # iteration duration
        duration = time.time() - before_op_time

        if self.is_checkpoint(step):
            self.log_time(epoch, batch_idx * world_size, duration, losses, start_time)
            self.log_tb(mode, inputs, outputs, losses, step)
                
    def is_checkpoint(self, step):
        """ 
        Log less frequently after the early phase steps
        """
        early_phase = (step % self.log_frequency == 0) and (step < self.early_phase)
        late_phase = step % self.late_log_frequency == 0
        return (early_phase or late_phase)

    def log_time(self, epoch, batch_idx, duration, loss, start_time):
        """
        This function prints epoch, iteration, duration, loss and spent time.
        """
        rep_loss = loss['total_loss'].item()
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - start_time
        print(f'epoch: {epoch:2d} | batch: {batch_idx:6d} |' + \
              f'examples/s: {samples_per_sec:5.1f} | loss: {rep_loss:.3f} | time elapsed: {pretty_ts(time_sofar)}')
        
    def log_tb(self, mode, inputs, outputs, losses, step):
        """
        This function logs outputs for monitoring using tensorboard.
        """
        writer = self.writers[mode]
        # loss
        for l, v in losses.items():
            writer.add_scalar(f'{l}', v, step)
        
        scale = 0 # plot the maximum scale
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            
            plot_tb(writer, step, inputs[('color', 0, scale)][:, cam_id, ...], set_tb_title('cam', cam_id)) # frame_id 0            
            plot_disp_tb(writer, step, target_view[('disp', scale)], set_tb_title('cam', cam_id, 'disp')) # disparity
            plot_tb(writer, step, target_view[('reproj_loss', scale)], set_tb_title('cam', cam_id, 'reproj')) # reprojection image
            plot_tb(writer, step, target_view[('reproj_mask', scale)], set_tb_title('cam', cam_id, 'reproj_mask')) # reprojection mask
            plot_tb(writer,  step, inputs['mask'][:, cam_id, ...], set_tb_title('cam', cam_id, 'self_occ_mask'))
    
            if self.spatio:
                plot_norm_tb(writer, step, target_view[('overlap', 0, scale)], set_tb_title('cam', cam_id, 'sp'))
                plot_tb(writer, step, target_view[('overlap_mask', 0, scale)], set_tb_title('cam', cam_id, 'sp_mask'))
                
            if self.spatio_temporal:
                for frame_id in self.frame_ids:
                    if frame_id == 0:
                        continue
                    plot_norm_tb(writer, step, target_view[('color', frame_id, scale)], set_tb_title('cam', cam_id, 'pred_', frame_id))                      
                    plot_norm_tb(writer, step, target_view[('overlap', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_', frame_id))
                    plot_tb(writer, step, target_view[('overlap_mask', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_mask_', frame_id))
                    
            if self.aug_depth:
                plot_disp_tb(writer, step, target_view[('disp', scale, 'aug')], set_tb_title('view_aug', cam_id))                

    def log_result(self, inputs, outputs, idx, syn_visualize=False):
        """
        This function logs outputs for visualization.
        """        
        scale = 0
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            disps = target_view['disp', scale]
            for jdx, disp in enumerate(disps):       
                disp = colormap(disp)[0,...].transpose(1,2,0)
                disp = pil.fromarray((disp * 255).astype(np.uint8))
                cur_idx = idx*self.batch_size + jdx 
                disp.save(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_disp.jpg'))
            
        if syn_visualize:    
            syn_disps = outputs['disp_vis']
            for kdx, syn_disp in enumerate(syn_disps):
                syn_disp = colormap(syn_disp)[0,...].transpose(1,2,0)
                syn_disp = pil.fromarray((syn_disp * 255).astype(np.uint8))
                syn_disp.save(os.path.join(self.syn_path, f'{kdx:03d}_syndisp.jpg'))
                
    def compute_depth_losses(self, inputs, outputs, vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth
        # img_mask = inputs['mask']


        med_scale = []
        
        error_metric_dict = defaultdict(float)
        error_median_dict = defaultdict(float)

        for cam in range(self.num_cams):
            # mask_pred = img_mask[:,cam,...]

            target_view = outputs['cam', cam]

            depth_gt = inputs['depth'][:, cam, ...]
            
            _, _, h, w = depth_gt.shape

            depth_pred = target_view[('depth', 0)].to(depth_gt.device)
            depth_pred = torch.clamp(F.interpolate(
                        depth_pred, [h, w], mode='bilinear', align_corners=False), 
                         min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()

            mask = (depth_gt > min_eval_depth) * (depth_gt < max_eval_depth) * inputs['mask'][:, cam, ...]
            mask = mask.bool()
            
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            # print("depth_shape: ",depth_gt.size(),depth_pred.size())
            # print("depth_mask: ",mask_pred.size())
            #
            # calculate median scale
            scale_val = torch.median(depth_gt) / torch.median(depth_pred)
            med_scale.append(round(scale_val.cpu().numpy().item(), 2))
                            
            depth_pred_metric = torch.clamp(depth_pred, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric = cal_depth_error(depth_pred_metric, depth_gt)
            
            depth_pred_median = torch.clamp(depth_pred * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median = cal_depth_error(depth_pred_median, depth_gt)


            # print("depth_errors_metric len: ", len(depth_errors_metric))
            for i in range(len(depth_errors_metric)):
                key = self._metric_names[i]
                error_metric_dict[key] += depth_errors_metric[i]
                error_median_dict[key] += depth_errors_median[i]

        if vis_scale==True:
            # print median scale
            print(f'          | median scale = {med_scale}')
        print("error_metric_dict.keys(): ",error_metric_dict.keys())
        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy() / self.num_cams
            error_median_dict[key] = error_median_dict[key].cpu().numpy() / self.num_cams
            
        return error_metric_dict, error_median_dict

    def compute_depth_losses_cam(self, inputs, outputs, save_del=False,vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth
        # img_mask = inputs['mask']

        med_scale = []

        error_metric_dict = defaultdict(float)
        error_median_dict = defaultdict(float)
        single_cam_dict = {f"cam{i}": None for i in range(6)}
        if save_del:
            abs_rel_met, sq_rel_met, rmse_met, rmse_log_met, abs_rel_med, sq_rel_med, rmse_med, rmse_log_med = [],[],[],[],[],[],[],[]
        else:
            abs_rel_met, sq_rel_met, rmse_met, rmse_log_met, abs_rel_med, sq_rel_med, rmse_med, rmse_log_med = [], [], [], [], [], [], [], []
        for cam in range(self.num_cams):
            # mask_pred = img_mask[:,cam,...]

            target_view = outputs['cam', cam]

            depth_gt = inputs['depth'][:, cam, ...]

            _, _, h, w = depth_gt.shape

            depth_pred = target_view[('depth', 0)].to(depth_gt.device)
            depth_pred = torch.clamp(F.interpolate(
                depth_pred, [h, w], mode='bilinear', align_corners=False),
                min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()
            # print("depth_pred is: ", type(depth_pred), depth_pred.size())
            # print("depth_gt is: ", type(depth_gt), depth_gt.size())
            # depth_pred is: <
            #
            # class 'torch.Tensor'> torch.Size([1, 1, 384, 640])
            #
            # depth_gt is: <
            #
            # class 'torch.Tensor'> torch.Size([1, 1, 384, 640])

            mask = (depth_gt > min_eval_depth) * (depth_gt < max_eval_depth) * inputs['mask'][:, cam, ...]
            mask = mask.bool()

            # print("save_del: ", save_del)
            if save_del:
                abs_rel_full_met_e, sq_rel_full_met_e, rmse_full_met_e, rmse_log_full_met_e = cal_depth_error_del(depth_pred, depth_gt,mask,min_eval_depth, max_eval_depth,type="met")
                abs_rel_full_med_e, sq_rel_full_med_e, rmse_full_med_e, rmse_log_full_med_e = cal_depth_error_del(depth_pred, depth_gt, mask,
                                                                                          min_eval_depth,
                                                                                          max_eval_depth, type="med")
                abs_rel_met.append(abs_rel_full_met_e)
                sq_rel_met.append(sq_rel_full_met_e)
                rmse_met.append(rmse_full_met_e)
                rmse_log_met.append(rmse_full_met_e)
                abs_rel_med.append(abs_rel_full_med_e)
                sq_rel_med.append(sq_rel_full_med_e)
                rmse_med.append(rmse_full_med_e)
                rmse_log_med.append(rmse_full_med_e)
                # print("sizes(): ", abs_rel_full.size(), sq_rel_full.size(), rmse_full.size(), rmse_log_full.size())
            else:
                pass
            # if save_del:
            #     result = cal_depth_error_del(depth_pred_metric, depth_gt)
            # print("MASK is: ", type(mask), mask.size()) torch.Size([1, 1, 384, 640]

            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            # print("depth_shape: ",depth_gt.size(),depth_pred.size()) torch.Size([41870]) torch.Size([41870])
            # print("depth_mask: ",mask_pred.size())
            #
            # calculate median scale
            scale_val = torch.median(depth_gt) / torch.median(depth_pred)
            med_scale.append(round(scale_val.cpu().numpy().item(), 2))

            depth_pred_metric = torch.clamp(depth_pred, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric = cal_depth_error(depth_pred_metric, depth_gt)


            # print("depth_errors_metric: ", type(depth_errors_metric), depth_errors_metric.size())

            depth_pred_median = torch.clamp(depth_pred * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median = cal_depth_error(depth_pred_median, depth_gt)
            # print("depth_errors_median: ", type(depth_errors_median), depth_errors_median.size())

            single_error_metric_dict = defaultdict(float)
            single_error_median_dict = defaultdict(float)

            # print("depth_errors_metric len: ", len(depth_errors_metric))
            for i in range(len(depth_errors_metric)):
                key = self._metric_names[i]
                error_metric_dict[key] += depth_errors_metric[i]
                error_median_dict[key] += depth_errors_median[i]

                single_error_metric_dict[key] = depth_errors_metric[i]
                single_error_median_dict[key] = depth_errors_median[i]
            single_cam_dict["cam" + str(cam)] = {"metric": single_error_metric_dict, "median": single_error_median_dict}

        if vis_scale == True:
            # print median scale
            print(f'          | median scale = {med_scale}')
        # print("error_metric_dict.keys(): ", error_metric_dict.keys())
        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy() / self.num_cams
            error_median_dict[key] = error_median_dict[key].cpu().numpy() / self.num_cams
            for cam in range(self.num_cams):
                single_cam_dict["cam" + str(cam)]['metric'][key] = single_cam_dict["cam" + str(cam)]['metric'][key].cpu().numpy()
                single_cam_dict["cam" + str(cam)]['median'][key] = single_cam_dict["cam" + str(cam)]['median'][key].cpu().numpy()
        if save_del:
            # print("--------length---:",len(abs_rel_met))
            abs_rel_met = torch.stack(abs_rel_met, dim=1).squeeze(2)
            # print("--------alength---:",abs_rel_met.size())
            sq_rel_met = torch.stack(sq_rel_met, dim=1).squeeze(2)
            rmse_met = torch.stack(rmse_met, dim=1).squeeze(2)
            rmse_log_met = torch.stack(rmse_log_met, dim=1).squeeze(2)
            abs_rel_med = torch.stack(abs_rel_med, dim=1).squeeze(2)
            sq_rel_med = torch.stack(sq_rel_med, dim=1).squeeze(2)
            rmse_med = torch.stack(rmse_med, dim=1).squeeze(2)
            rmse_log_med = torch.stack(rmse_log_med, dim=1).squeeze(2)
            return error_metric_dict, error_median_dict,single_cam_dict,[abs_rel_met, sq_rel_met, rmse_met, rmse_log_met, abs_rel_med, sq_rel_med, rmse_med, rmse_log_med]
        else:
            return error_metric_dict, error_median_dict,single_cam_dict


    def print_perf(self, loss, scale): 
        """
        This function prints various metrics for depth estimation accuracy.
        """
        perf = ' '*3 + scale
        for k, v in loss.items():
            perf += ' | ' + str(k) + f': {v:.3f}'
        print(perf)
            