# Built-in function
import os, sys
from datetime import datetime
import copy
import json
import random
import time
import math
from collections import deque
import argparse


# utils library
import numpy as np
import imageio
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


import curriculums
from model import siren

import matplotlib.pyplot as plt

from torch_ema import ExponentialMovingAverage

""" this code use data parellel processing with pytorch DDP framwork.
inference : https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html
"""


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)
    
    # trainer
    parser.add_argument("--n_epochs",type=int,default=3000,help="number of epochs of training") # change rather 
    return parser

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z

def train(rank, world_size, opt):
    torch.manual_seed(0)
    
    # device setting
    setup(rank, world_size, opt.port)
    device = torch.device(rank)
    
    # curriculum extracting
    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    
    # model declaration
    SIREN = getattr(siren, metadata['Radiance_Generator'])
    SIREN=SIREN()
    
    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    

if __name__ == '__main__':
    
    opt = config_parser()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)