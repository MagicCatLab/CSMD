# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
import torch

def pack_cam_feat(x):
    """
    This function packs camera dimension to batch dimension.
    """
    if isinstance(x, dict):
        for k, v in x.items():
            b, n_cam = v.shape[:2]
            x[k] = v.view(b*n_cam, *v.shape[2:])
        return x
    else:
        b, n_cam = x.shape[:2]
        x = x.view(b*n_cam, *x.shape[2:])
    return x


def conv2d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, nonlin='LRU', padding_mode='reflect',
           norm=False):
    """
    This function computes 2d convolutions followed by the specific nonlinear function(LeakyReLU, ELU, or None)
    """

    if nonlin == 'LRU':
        act = nn.LeakyReLU(0.1, inplace=True)
    elif nonlin == 'ELU':
        act = nn.ELU(inplace=True)
    else:
        act = nn.Identity()

    if norm:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=((kernel_size - 1) * dilation) // 2, bias=False, padding_mode=padding_mode)
        bnorm = nn.BatchNorm2d(out_planes)
    else:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        bnorm = nn.Identity()
    return nn.Sequential(conv, bnorm, act)

def unpack_cam_feat(x, b, n_cam):
    """
    This function unpacks batch dimension into batch and camera dimensions.
    """
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = v.view(b, n_cam, *v.shape[1:])
        return x
    else:
        x = x.view(b, n_cam, *x.shape[1:])
    return x


def conv1d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, nonlin='LRU', padding_mode='reflect',
           norm=False):
    """
    This function computes 1d convolutions follwed by the nonlinear function(LeakyReLU, or None)
    """
    if nonlin == 'LRU':
        act = nn.LeakyReLU(0.1, inplace=True)
    elif nonlin == 'ELU':
        act = nn.ELU(inplace=True)
    else:
        act = nn.Identity()

    if norm:
        conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=((kernel_size - 1) * dilation) // 2, bias=False, padding_mode=padding_mode)
        bnorm = nn.BatchNorm1d(out_planes)
    else:
        conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        bnorm = nn.Identity()

    return nn.Sequential(conv, bnorm, act)

def upsample(x):
    """
    This function upsamples input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode='nearest')

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
