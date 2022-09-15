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
                          torch.linspace(1, -1, H, device=device)) # x,y grid , it descrete -1 to 1 with W resolution
    x = x.T.flatten() # transform하면 적절하게 x좌표에 맞는다.
    y = y.T.flatten() #  transform하면 잘 맞는다.
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2) 

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1)) # x,y,z 좌표. 정규화됨. (h*w,3)

    # (h*w,num_steps,1) : 시작, 끝을 n개로 나누고 그걸 w*h크기만큼 반복.
    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals #[h*w,num_steps,3] : 크기,step,좌표 : ray의 near,far uniform sample 좌표

    points = torch.stack(n*[points]) # 배치개만큼 만들어냄 (n,h*w,num_steps,coords:3)
    z_vals = torch.stack(n*[z_vals]) # near_far 간격을 배치개만큼 (n,h*w,num_steps,1)
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device) # (n,h*w,3) : 배치,크기,좌표

    return points, z_vals, rays_d_cam # 출력으로 ray의 등간격 Sample point, 각 등간격 범위, 시작 ray 좌표

def transform_sampled_points(points, z_vals, ray_directions, device, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal'):
    """Samples a camera position and maps points in camera space to world space."""

    n, num_rays, num_steps, channels = points.shape

    points, z_vals = perturb_points(points, z_vals, ray_directions, device)


    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)


    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw

def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta


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

def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""

    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1-torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights

def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


