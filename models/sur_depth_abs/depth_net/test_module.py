from collections import OrderedDict

import torch
import torch.nn as nn
from blocks import *



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
        input_features[0] = input_features[0].view(-1,64, 192, 320*6)
        input_features[1] = input_features[1].view(-1,64, 96, 160*6)
        input_features[2] = input_features[2].view(-1,128, 48, 80*6)

        print("inputs: ",len(input_features),input_features[0].size(),input_features[1].size(),input_features[2].size())
        print(self.scales ,
        self.use_skips ,
        self.level_in,
        self.num_ch_enc,
        self.num_ch_dec)

        # decode
        x = input_features[-1]
        print("self.level in is: ",self.level_in)
        print("use_skip: ", self.use_skips)
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))
                print("self.outputs[(disp, i)]: ", outputs[("disp", i)].size())
                b,c,w,h = outputs[('disp', i)].size()
                outputs[('disp', i)] = outputs[('disp', i)].view(b*6,c,w,-1)
        return outputs


if __name__ == "__main__":
    model = DepthDecoder(2,  [ 64 ,64, 128] ,  [16, 32, 64, 128, 256], [0], False)
    input_features = [
        torch.randn(6, 64, 192, 320),
        torch.randn(6, 64, 96, 160),
        torch.randn(6, 128, 48, 80)
    ]

    # 将输入数据传递给模型
    outputs = model(input_features)

    # 打印输出的结果
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")