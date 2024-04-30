'''
这个depth_net分为两种形态构成，
(1) resnet+voxfusion+cvt*4+depth_decoder
(2) resnet+voxfusion+voxcvt*4+depth_decoder
'''

# Copyright (c) 2023 42dot. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_component.resnet_encoder import ResnetEncoder
from .blocks import *
from .vox_fusion import VFNet
from .CVT import CVT

class VIT_VF_DepthNet(nn.Module):
    """
    Depth fusion module
    """

    def __init__(self, cfg, rank):
        super(VIT_VF_DepthNet, self).__init__()
        self.read_config(cfg)

        # feature encoder
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init,
                                     1)  # number of layers, pretrained, number of input images
        self.num_ch_enc = self.encoder.num_ch_enc
        del self.encoder.encoder.fc
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:])
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode='reflect')

        fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level]
        self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model='depth')

        num_ch_enc = self.encoder.num_ch_enc[:(self.fusion_level + 1)]
        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(self.fusion_level, num_ch_enc, num_ch_dec, self.scales, use_skips=self.use_skips)
        self.cross = {}

        for i in range(len(self.num_ch_enc)-2):
            self.cross[i] = CVT(input_channel=self.num_ch_enc[i], downsample_ratio=2**(len(self.num_ch_enc) -1 - i), iter_num=self.iter_num[i]).to(device=torch.device('cuda:'+str(rank)))
            # print("CVT models in device: ")


    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def forward(self, inputs):
        outputs = {}

        # dictionary initialize
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        lev = self.fusion_level

        # packed images for surrounding view
        sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        packed_input = pack_cam_feat(sf_images)
        packed_feats = self.encoder(packed_input)
        # print("pack_featd: ", len(packed_feats), packed_feats[0].size(), packed_feats[1].size(), packed_feats[2].size(), packed_feats[3].size(), packed_feats[4].size())

        _, _, up_h, up_w = packed_feats[lev].size()
        packed_feats_list = packed_feats[lev:lev + 1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in
                               packed_feats[lev + 1:]]

        # print("packed_feats_list: ", len(packed_feats_list), packed_feats_list[1].size(), packed_feats_list[0].size(),packed_feats_list[2].size())
        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))
        # print("packed_feats_agg: ", packed_feats_agg.size())
        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams)



        # print("feat_agg", type(feats_agg), feats_agg.size())


        fusion_dict = self.fusion_net(inputs, feats_agg)

        fusion_feats = [fusion_dict['proj_feat']]
        # print("fusion_feats； ", type(fusion_feats), len(fusion_feats), feats_agg[0].size())

        feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]
        # feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']] + packed_feats[lev+1:]
        # print("feat_in: ",len(feat_in), feat_in[0].size(), feat_in[1].size(), feat_in[2].size())

        feat_cvt_in = []
        for i in range(len(feat_in)):
            B, C, H, W = feat_in[i].size()

            fi = feat_in[i].view(-1, 6, C, H, W)
            # print("fio: ",fi.size())
            fi = fi.view(-1,1,C,H,W*6)
            # print()
            # print("fi: ", fi.size())
            tra_feat = self.cross[i](fi).view(B, C, H, W)
            # print("tra_feat size: ", tra_feat.size())
            # tra_feat = tra_feat
            feat_cvt_in.append(feat_in[i]+tra_feat)
        # print("feat_cvt_in:", len(feat_cvt_in), feat_cvt_in[0].size(), feat_cvt_in[1].size(), feat_cvt_in[2].size() )

        packed_depth_outputs = self.decoder(feat_cvt_in)

        depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams)

        for cam in range(self.num_cams):
            for k in depth_outputs.keys():
                outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
                # print(depth_outputs[k][:, cam, ...].size())


        return outputs


class DepthDecoder(nn.Module):
    """
    This class decodes encoded 2D features to estimate depth map.
    Unlike monodepth depth decoder, we decode features with corresponding level we used to project features in 3D (default: level 2(H/4, W/4))
    """

    def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2), use_skips=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1
        self.scales = scales
        self.use_skips = use_skips

        self.level_in = level_in
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        self.convs = OrderedDict()
        for i in range(self.level_in, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin='ELU')

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin='ELU')

        for s in self.scales:
            self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin=None)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        # decode
        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))
                # print("self.outputs[(disp, i)]: ", outputs[("disp", i)].size())
        return outputs