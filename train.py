import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm, trange
import argparse

import matplotlib.pyplot as plt

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

def train():
    torch.manual_seed(0)
    
    setup(rank, world_size, opt.port)
    device = torch.device(rank)
    

if __name__ == '__main__':
    
    opt = config_parser()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)