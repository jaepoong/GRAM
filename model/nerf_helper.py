import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from utils_.math_utils import *

def get_rays(H, W, K, c2w):
    # 좌표 그리드 생성
    # week perspective camera 좌표계 추출.
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) # intrinsic
    # Rotate ray directions from camera frame to the world frame
    # intrinsic * extrinsic
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape) # extrinsic- transform  * len(ray_d)
    return rays_o, rays_d

def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device)) # x,y grid
    x = x.T.flatten() # transform하면 적절하게 x좌표에 맞는다.
    y = y.T.flatten() #  transform하면 잘 맞는다.
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))


    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam


def truncated_normal_(tensor, mean=0, std=1):
    """ tensor trunction 진행

    Args:
        tensor (tensor): input tensor
        mean (int, optional): normal distribution mean for truncation. Defaults to 0.
        std (int, optional): normal distribution std for truncation. Defaults to 1.

    Returns:
        tensor: truncated tensor 
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2) # -2<tmp<2 사이인지? boolean
    ind = valid.max(-1, keepdim=True)[1] # 
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor 