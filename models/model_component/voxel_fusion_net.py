import torch
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models.model_component.norm_type import *


class Vox_fusion_net(nn.Module):
    def __init__(self,cfg , rank):
        super(Vox_fusion_net, self).__init__()
        self.read_config(cfg)
        self.eps = 1e-8
        self.voxel_end_p = [self.voxel_str_p[i] + self.voxel_unit_size[i] * (self.voxel_size[i] - 1) for i in range(3)]
        # define a voxel space, [1, 3, z, y, x], each voxel contains its 3D position
        voxel_grid = self.create_voxel_grid(self.voxel_str_p, self.voxel_end_p, self.voxel_size)
        b, _, self.z_dim, self.y_dim, self.x_dim = voxel_grid.size()
        self.n_voxels = self.z_dim * self.y_dim * self.x_dim
        ones = torch.ones(self.batch_size, 1, self.n_voxels)
        self.voxel_pts = torch.cat([voxel_grid.view(b, 3, self.n_voxels), ones], dim=1).cuda()
        # print("self.voxel_pts is:", self.voxel_pts.device)

    # voxel - preprocessing layer
        self.v_dim_o = [(self.fusion_feat_in_dim + 1) * 2] + self.voxel_pre_dim
        self.v_dim_no = [self.fusion_feat_in_dim + 1] + self.voxel_pre_dim

        self.conv_overlap = conv1d(self.v_dim_o[0], self.v_dim_o[1], kernel_size=1)
        self.conv_non_overlap = conv1d(self.v_dim_no[0], self.v_dim_no[1], kernel_size=1)

        encoder_dims = self.proj_d_bins * self.v_dim_o[-1]
        stride = 1


    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def create_voxel_grid(self, str_p, end_p, v_size):
        """
        output: [batch, 3, z_dim, y_dim, x_dim]
        [b, :, z, y, x] contains (x,y,z) 3D point
        """
        grids = [torch.linspace(str_p[i], end_p[i], v_size[i]) for i in range(3)]

        x_dim, y_dim, z_dim = v_size
        grids[0] = grids[0].view(1, 1, 1, 1, x_dim)
        grids[1] = grids[1].view(1, 1, 1, y_dim, 1)
        grids[2] = grids[2].view(1, 1, z_dim, 1, 1)

        grids = [grid.expand(self.batch_size, 1, z_dim, y_dim, x_dim) for grid in grids]
        return torch.cat(grids, 1)

    def calculate_sample_pixel_coords(self, K, v_pts, w_dim, h_dim):
        """
        This function calculates pixel coords for each point([batch, n_voxels, 1, 2]) to sample the per-pixel feature.
        """
        cam_points = torch.matmul(K[:, :3, :3], v_pts)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        if not torch.all(torch.isfinite(pix_coords)):
            pix_coords = torch.clamp(pix_coords, min=-w_dim*2, max=w_dim*2)

        pix_coords = pix_coords.view(self.batch_size, 2, self.n_voxels, 1)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (w_dim - 1)
        pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (h_dim - 1)
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def calculate_valid_mask(self, mask_img, pix_coords, v_pts_local):
        """
        This function creates valid mask in voxel coordinate by projecting self-occlusion mask to 3D voxel coords.
        """
        # compute validity mask, [b, 1, n_voxels, 1]
        mask_selfocc = (F.grid_sample(mask_img, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True) > 0.5)
        # discard points behind the camera, [b, 1, n_voxels]
        mask_depth = (v_pts_local[:, 2:3, :] > 0)
        # compute validity mask, [b, 1, n_voxels, 1]
        pix_coords_mask = pix_coords.permute(0, 3, 1, 2)
        mask_oob = ~(torch.logical_or(pix_coords_mask > 1, pix_coords_mask < -1).sum(dim=1, keepdim=True) > 0)
        valid_mask = mask_selfocc.squeeze(-1) * mask_depth * mask_oob.squeeze(-1)
        return valid_mask

    def preprocess_non_overlap(self, voxel_feat_list, voxel_mask_list, voxel_mask_count):
        """
        This function applies 1x1 convolutions to features from non-overlapping features.
        """
        non_overlap_mask = (voxel_mask_count == 1)
        voxel = sum(voxel_feat_list)
        voxel = voxel * non_overlap_mask.float()

        for conv_no in self.conv_non_overlap:
            voxel = conv_no(voxel)
        return voxel * non_overlap_mask.float()

    def preprocess_overlap(self, voxel_feat_list, voxel_mask_list, voxel_mask_count):
        """
        This function applies 1x1 convolutions on overlapping features.
        Camera configuration [0,1,2] or [0,1,2,3,4,5]:
                        3 1
            rear cam <- 5   0 -> front cam
                        4 2
        """
        overlap_mask = (voxel_mask_count == 2)
        if self.num_cams == 3:
            feat1 = voxel_feat_list[0]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2]
        elif self.num_cams == 6:
            feat1 = voxel_feat_list[0] + voxel_feat_list[3] + voxel_feat_list[4]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2] + voxel_feat_list[5]
        else:
            raise NotImplementedError

        voxel = torch.cat([feat1, feat2], dim=1)
        for conv_o in self.conv_overlap:
            voxel = conv_o(voxel)
        return voxel * overlap_mask.float()

    def backproject_into_voxel(self, feats_agg, input_mask, intrinsics, extrinsics):
        """
        This function backprojects 2D features into 3D voxel coordinate using intrinsic and extrinsic of each camera.
        Self-occluded regions are removed by using the projected mask in 3D voxel coordinate.
        """
        voxel_feat_list = []
        voxel_mask_list = []

        for cam in range(self.num_cams):
            feats_img = feats_agg[:, cam, ...]
            _, _, h_dim, w_dim = feats_img.size()

            mask_img = input_mask[:, cam, ...]
            mask_img = F.interpolate(mask_img, [h_dim, w_dim], mode='bilinear', align_corners=True)

            # 3D points in the voxel grid -> 3D points referenced at each view. [b, 3, n_voxels]
            ext_inv_mat = extrinsics[:, cam, :3, :]

            v_pts_local = torch.matmul(ext_inv_mat, self.voxel_pts)
            # print("v_ptslocal ",v_pts_local.dtype)




            # calculate pixel coordinate that each point are projected in the image. [b, n_voxels, 1, 2]
            K_mat = intrinsics[:, cam, :, :]
            pix_coords = self.calculate_sample_pixel_coords(K_mat, v_pts_local, w_dim, h_dim)

            # compute validity mask. [b, 1, n_voxels]
            valid_mask = self.calculate_valid_mask(mask_img, pix_coords, v_pts_local)

            # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
            feat_warped = F.grid_sample(feats_img, pix_coords, mode='bilinear', padding_mode='zeros',
                                        align_corners=True)
            # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
            feat_warped = torch.cat([feat_warped.squeeze(-1), v_pts_local[:, 2:3, :] / (self.voxel_size[0])], dim=1)
            feat_warped = feat_warped * valid_mask.float()

            voxel_feat_list.append(feat_warped)
            voxel_mask_list.append(valid_mask)

        # compute overlap region
        voxel_mask_count = torch.sum(torch.cat(voxel_mask_list, dim=1), dim=1, keepdim=True)
            # discriminatively process overlap and non_overlap regions using different MLPs
        voxel_non_overlap = self.preprocess_non_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
        voxel_overlap = self.preprocess_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
        voxel_feat = voxel_non_overlap + voxel_overlap

        return voxel_feat

    def forward(self, inputs, multi_img_feat):
        mask = inputs["mask"]
        K = inputs['K', self.fusion_level+1]
        inv_K = inputs['inv_K', self.fusion_level+1]
        extrinsics = inputs['extrinsics']
        extrinsics_inv = inputs['extrinsics_inv']
        # print(mask.dtype, K.dtype, inv_K.dtype, extrinsics_inv.dtype)

        fusion_dict = {}
        for cam in range(self.num_cams):
            fusion_dict[('cam', cam)] = {}

        sample_tensor = multi_img_feat[0, 0, ...] # B, n_cam, c, h, w

        voxel_feat = self.backproject_into_voxel(multi_img_feat, mask, K, extrinsics)

        return voxel_feat
