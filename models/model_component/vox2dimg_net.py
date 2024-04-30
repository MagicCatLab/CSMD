import torch
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models.model_component.norm_type import *


class Vox_dimg_net(nn.Module):
    def __init__(self,cfg , rank):
        super(Vox_dimg_net, self).__init__()
        self.read_config(cfg)
        self.eps = 1e-8
        self.voxel_end_p = [self.voxel_str_p[i] + self.voxel_unit_size[i] * (self.voxel_size[i] - 1) for i in range(3)]
        # # define a voxel space, [1, 3, z, y, x], each voxel contains its 3D position
        voxel_grid = self.create_voxel_grid(self.voxel_str_p, self.voxel_end_p, self.voxel_size)
        b, _, self.z_dim, self.y_dim, self.x_dim = voxel_grid.size()
        self.n_voxels = self.z_dim * self.y_dim * self.x_dim
        # ones = torch.ones(self.batch_size, 1, self.n_voxels)
        # self.voxel_pts = torch.cat([voxel_grid.view(b, 3, self.n_voxels), ones], dim=1)

        # define grids in pixel space
        self.img_h = self.height // (2 ** (self.fusion_level + 1))
        self.img_w = self.width // (2 ** (self.fusion_level + 1))
        self.num_pix = self.img_h * self.img_w
        self.pixel_grid = self.create_pixel_grid(self.batch_size, self.img_h, self.img_w).cuda()


        self.pixel_ones = torch.ones(self.batch_size, 1, self.proj_d_bins, self.num_pix).cuda()

        # define a depth grid for projection
        depth_bins = torch.linspace(self.proj_d_str, self.proj_d_end, self.proj_d_bins)
        self.depth_grid = self.create_depth_grid(self.batch_size, self.num_pix, self.proj_d_bins, depth_bins).cuda()


        # voxel - preprocessing layer
        self.v_dim_o = [(self.fusion_feat_in_dim + 1) * 2] + self.voxel_pre_dim
        self.v_dim_no = [self.fusion_feat_in_dim + 1] + self.voxel_pre_dim

        self.conv_overlap = conv1d(self.v_dim_o[0], self.v_dim_o[1], kernel_size=1)
        self.conv_non_overlap = conv1d(self.v_dim_no[0], self.v_dim_no[1], kernel_size=1)

        encoder_dims = self.proj_d_bins * self.v_dim_o[-1]
        print(encoder_dims)
        stride = 1

        self.reduce_dim = nn.Sequential(*conv2d(encoder_dims, 256, kernel_size=3, stride=stride).children(),
                                        *conv2d(256, self.bpro_feat_out_dim, kernel_size=3, stride=stride).children())

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

    def create_pixel_grid(self, batch_size, height, width):
        """
        output: [batch, 3, height * width]
        """
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, axis=0).unsqueeze(0).view(1, 2, height * width)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        ones = torch.ones(batch_size, 1, height * width)
        pix_coords = torch.cat([pix_coords, ones], 1)
        return pix_coords

    def create_depth_grid(self, batch_size, n_pixels, n_depth_bins, depth_bins):
        """
        output: [batch, 3, num_depths, height * width]
        """
        depth_layers = []
        for d in depth_bins:
            depth_layer = torch.ones((1, n_pixels)) * d
            depth_layers.append(depth_layer)
        depth_layers = torch.cat(depth_layers, dim=0).view(1, 1, n_depth_bins, n_pixels)
        depth_layers = depth_layers.expand(batch_size, 3, n_depth_bins, n_pixels)
        return depth_layers

    def project_voxel_into_image(self, voxel_feat, inv_K, extrinsics_inv):
        """
        This function projects voxels into 2D image coordinate.
        [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        """
        # define depth bin
        # [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        b, feat_dim, _ = voxel_feat.size()
        voxel_feat = voxel_feat.view(b, feat_dim, self.z_dim, self.y_dim, self.x_dim)

        proj_feats = []
        for cam in range(self.num_cams):
            # construct 3D point grid for each view
            cam_points = torch.matmul(inv_K[:, cam, :3, :3], self.pixel_grid)
            cam_points = self.depth_grid * cam_points.view(self.batch_size, 3, 1, self.num_pix)
            cam_points = torch.cat([cam_points, self.pixel_ones], dim=1)  # [b, 4, n_depthbins, n_pixels]
            cam_points = cam_points.view(self.batch_size, 4, -1)  # [b, 4, n_depthbins * n_pixels]

            # apply extrinsic: local 3D point -> global coordinate, [b, 3, n_depthbins * n_pixels]
            points = torch.matmul(extrinsics_inv[:, cam, :3, :], cam_points)

            # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
            grid = points.permute(0, 2, 1)

            for i in range(3):
                v_length = self.voxel_end_p[i] - self.voxel_str_p[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_str_p[i]) / v_length * 2. - 1.

            grid = grid.view(self.batch_size, self.proj_d_bins, self.img_h, self.img_w, 3)
            proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            proj_feat = proj_feat.view(b, self.proj_d_bins * self.v_dim_o[-1], self.img_h, self.img_w)

            # conv, reduce dimension
            proj_feat = self.reduce_dim(proj_feat)
            proj_feats.append(proj_feat)
        return proj_feats

    def forward(self, inputs, vox_map):
        mask = inputs["mask"]
        K = inputs['K', self.fusion_level+1]
        inv_K = inputs['inv_K', self.fusion_level+1]
        extrinsics = inputs['extrinsics']
        extrinsics_inv = inputs['extrinsics_inv']

        dimg_feat = self.project_voxel_into_image(vox_map, inv_K, extrinsics_inv)
        return dimg_feat