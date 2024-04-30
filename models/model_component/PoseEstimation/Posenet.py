import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_component.resnet_encoder import ResnetEncoder
from collections import OrderedDict
from models.encoder import Pose_ResnetEncoder

class IF_PoseNet(nn.Module):
    def __init__(self, cfg):
        super(IF_PoseNet, self).__init__()
        self.read_config(cfg)
        # self.pfeat_encoder = ResnetEncoder()
        self.Pose_encoder = Pose_ResnetEncoder(self.pose_encoder_layer,
                self.weights_init,
                num_input_images=self.num_pose_frames)

        self.num_ch_enc = self.Pose_encoder.num_ch_enc
        self.num_input_features = 1
        self.num_frames_to_predict_for = 2
        self.stride=1

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(self.num_input_features * 256, 256, 3, self.stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, self.stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * self.num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

        self.net = nn.ModuleList(list(self.convs.values()))




    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def forward(self, inputs, frame_id):
        sf_images_cur = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        sf_images_pre = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        # print("sf_images_cur is:", sf_images_cur.size())
        # print("sf_images_pre is:", sf_images_pre.size())
        # print("sf_images_next is:", sf_images_next.size())

        B, N, C, H, W = sf_images_cur.size()
        # print("sf_images_cur is:", sf_images_cur.size())
        if frame_id < 0:
            pose_inputs = [sf_images_pre.view(-1,C,H,W), sf_images_cur.view(-1,C,H,W)]
        else:
            pose_inputs = [sf_images_cur.view(-1,C,H,W), sf_images_next.view(-1,C,H,W)]
        pose_inputs = [self.Pose_encoder(torch.cat(pose_inputs, 1))]
        # print("pose input is: ", len(pose_inputs[0]), pose_inputs[0][0].size())

        B, C, H, W = pose_inputs[0][-1].shape

        if self.joint_pose:
            last_features = [f[-1].reshape(-1, 6, C, H, W).mean(1) for f in pose_inputs]
        else:
            last_features = [f[-1] for f in pose_inputs]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features

        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        #
        # print(axisangle.size(),translation.size())
        return axisangle, translation
