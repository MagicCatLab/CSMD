import torch
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models.model_component.norm_type import *

from models.model_component.norm_type import *
from models.model_component.resnet_encoder import ResnetEncoder
from models.model_component.voxel_fusion_net import Vox_fusion_net
from models.model_component.vox2dimg_net import Vox_dimg_net
from models.model_component.DepthEstimation.Solo import CVT2d
from models.model_component.DepthEstimation.depth_decoder import DepthDecoder
from collections import OrderedDict


class Depth_EstimationNet(nn.Module):
    def __init__(self,cfg,rank):
        super(Depth_EstimationNet, self).__init__()
        self.read_config(cfg)
        self.vox_fusion = Vox_fusion_net(cfg, rank)
        self.vox2dimg = Vox_dimg_net(cfg, rank)

        self.fusion_level_start_dim = sum(self.fusion_level_list[self.fusion_level:])

        self.conv2d = conv2d(self.fusion_level_start_dim, self.fusion_feat_in_dim, kernel_size=1,
                             padding_mode='reflect')

        # self.CVT_solo = []
        # for i in range(self.fusion_level + 1):
        #     self.CVT_solo.append(CVT2d(input_channel=self.fusion_level_list[i], downsample_ratio=2 ** (5 - 1 - i),
        #                                iter_num=self.CVT_iter_num))

        self.cvt0 = CVT2d(input_channel=self.fusion_level_list[0], downsample_ratio=2 ** (5 - 1 - 0),
                                       iter_num=self.CVT_iter_num)
        self.cvt1 = CVT2d(input_channel=self.fusion_level_list[1], downsample_ratio=2 ** (5 - 1 - 1),
                                       iter_num=self.CVT_iter_num)
        self.cvt2 = CVT2d(input_channel=self.fusion_level_list[2], downsample_ratio=2 ** (5 - 1 - 2),
                                       iter_num=self.CVT_iter_num)

        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(self.fusion_level, self.fusion_level_list[:self.fusion_level + 1], num_ch_dec,
                                    self.scales, use_skips=self.use_skips)

    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def forward(self, inputs, packed_feats):
        lev = self.fusion_level
        _, _, up_h, up_w = packed_feats[lev].size()

        packed_feats_list = packed_feats[lev:lev + 1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in
                               packed_feats[lev + 1:]]
        # concated_feature = torch.cat(packed_feats_list, dim=1)
        multi_img_feat = self.conv2d(torch.cat(packed_feats_list, dim=1))
        feats_agg = unpack_cam_feat(multi_img_feat, self.batch_size, self.num_cams)

        vox_feat = self.vox_fusion(inputs, feats_agg)
        dimg_deat = self.vox2dimg(inputs, vox_feat)

        img_concat =  torch.stack(dimg_deat,dim=1)


        con_feature = []
        # print("or shape cvt is:", packed_feats[0].size())

        B, C, H, W = packed_feats[0].size()
        cvt_feat0 = self.cvt0(packed_feats[0].view(-1,6,C,H,W))
        # print("shape cvt is:", packed_feats[0].size(),  cvt_feat0.size())
        con_feature.append(cvt_feat0)
        B, C, H, W = packed_feats[1].size()
        cvt_feat1 = self.cvt1(packed_feats[1].view(-1,6,C,H,W))

        # print("cvt_feat1 shape is:", cvt_feat1.size())
        con_feature.append(cvt_feat1)
        # for i in range(self.fusion_level):
        #     con_feature.append(self.CVT_solo[i](packed_feats[i].unsqueeze(0)))
        # trough_cvt_feature = self.CVT_solo[self.fusion_level](img_concat).view(-1, C, H, W)

        B,N,C,H,W = img_concat.size()
        trough_cvt_feature = self.cvt2(img_concat).view(-1, C, H, W)
        # print("trough_cvt_feature is: ", trough_cvt_feature.size())
        con_feature.append(trough_cvt_feature)
        outputs_depth = self.decoder(con_feature)
        return outputs_depth