import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # loader
    parser.add_argument("--data_dir",type=str,default="./data",
                        help=" input data directory")
    parser.add_argument("--output_dir",type=str,default=".logs/",
                        help="output ckpts and logs directory")
    

    
    # trainer
    parser.add_argument("--n_epochs",type=int,default=3000,help="number of epochs of training") # change rather
    
    
    return parser
    

def train():
    parser=config_parser()