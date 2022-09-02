"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
import time
import curriculums
from torch.cuda.amp import autocast

class Generator(nn.Module):
    def __init__(self,siren,z_dim,**kwargs):
        super().__init()
        self.z_dim=z_dim
        self.siren=siren(output_dim=4,z_dim=self.z_dim,input_dim=3)
        self.epoch=0
        self.step=0
    
    def set_device(self,device):
        self.device=device
        self.siren.device=device
        
        self.generate_avg_frequencies()
    
    def forward(self,z):
        batch_size=z.shape(0)
        
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, 
                                                                   num_steps, resolution=(img_size, img_size), 
                                                                   device=self.device, fov=fov, ray_start=ray_start, 
                                                                   ray_end=ray_end) # batch_size, pixels, num_steps, 1
            
        
        

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts
    
    