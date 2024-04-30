import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = []
        self.gradients = []

        # 注册hook
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, inputs, target_index=None):
        self.model.zero_grad()
        output = self.model(inputs)

        if target_index is None:
            target_index = output.argmax(dim=1)

        target = output[0, target_index]
        target.backward()

        gradients = self.gradients
        feature_maps = self.feature_maps
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = weights.mul(feature_maps).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=(inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)
        return cam