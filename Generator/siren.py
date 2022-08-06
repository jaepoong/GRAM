from operator import index
from re import X
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    """ film sine init"""
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class CustomMappingNetwork(nn.Module):
    """ this network is mapping network
    that make affine transform for siren gamma and beta~^

    Args:
        z_dim (int) : Gaussian noise dimension
        map_hidden_dim(int) : size of hidden Linear Perceptron
        map_output_dim(int) : size of output Linear Perceptron
    
    Output:
        frequencies ex)(batch,256)
        phase_shifts ex)(batch,256)
    """
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()



        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


def frequency_init(freq):
    """ frequency uniform init -> make random uniform frequency"""
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)
    
class ToOccupancy(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.layer=nn.Linear(input_dim,1)
    
    def forward(self,input,skip=None):
        out=self.layer(input)
        if skip is not None:
            out+=skip
        return out

class ToRGB(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.layer=nn.Linear(input_dim,3)
    
    def forward(self,input,skip=None):
        out=self.layer(input)
        if skip is not None:
            out=out+skip
        return out

class Radiance_Generator(nn.Module):
    """ radiance_generator
    입력으로 input,z,ray_directions를 받는다.
    z를 mappingnetwork를 통과해서 frequency와 phase_shifts를 뽑아낸다.
    skip_connection 구조를 사용해서 occupancy와 RGB값을 뽑아내는 네트워크이다.

    Args:
        input_dim (int): layer 입력의 dimension이다. intersection 좌표값이므로 (x,y,z)가 유력하다.
        z_dim (int) : gaussian latent의 크기이다.
        hidden_dim (int) : 전체적인 네트워크 generator상의 unit 개수이다.
        num_layer (int) : generator Network의 layer 개수
        
    """
    def ___init__(self, input_dim=3, z_dim=100, hidden_dim=256, device=None,num_layers=9):
        super().__init__()
        self.device=device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers=num_layers
        # modulist
        self.layer=nn.ModuleList()
        self.to_occupancy=nn.ModuleList()
        self.to_RGB=nn.ModuleList()
        
        # 첫번째 레이어
        self.layer.append(FiLMLayer(self.input_dim,self.hidden_dim))
        self.to_occupancy.append(ToOccupancy(self.hidden_dim))
        self.to_RGB.append(ToRGB(self.hidden_dim))
        
        # 마지막레이어 -1 까지
        for i in range(num_layers-3):
            self.layer.append(FiLMLayer(self.hidden_dim,self.hidden_dim))
            self.to_occupancy(ToOccupancy(self.hidden_dim))
            self.to_RGB(ToRGB(self.hidden_dim))
        
        self.layer.append(FiLMLayer(self.hidden_dim,self.hidden_dim))
        self.to_occupancy.append(ToOccupancy(self.hidden_dim))# 출력 occupancy
        self.layer.append(FiLMLayer(self.hidden_dim+3,self.hidden_dim))
        self.to_RGB.append(ToRGB(self.hidden_dim))
        
        self.mapping_network = CustomMappingNetwork(self.z_dim, self.hidden_dim, num_layers*hidden_dim*2) # mapping network
        
        # initiaizing
        self.layer.apply(frequency_init(25))
        self.layer[0].apply(first_layer_film_sine_init)
        self.to_RGB.apply(kaiming_leaky_init)
        self.to_occupancy.apply(kaiming_leaky_init)
    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z) # (9,256),(9,256)
        return self.forward_with_frequencies_phase_shifts(input,frequencies,phase_shifts,ray_directions,**kwargs)
                
    def forward_with_frequencies_phase_shifts(self,input,frequencies,phase_shifts,ray_directions,**kwargs):
        frequencies=frequencies*15+30
        rgb=0
        occupancy=0
        x=input
        
        for i in range(self.num_layers-2):
            start=i*self.hidden_dim
            end=(i+1)*self.hidden_dim
            x=self.layer[i](x,frequencies[...,start:end],phase_shifts[...,start:end])
            rgb=self.to_RGB[i](x,rgb)
            occupancy=self.to_occupancy(x,occupancy)
        ### 마지막 이전 skip부분 occypancy는 마지막에서 두번째 Film까지 accumulate
        start=start+self.hidden_dim
        end=end+self.hidden_dim
        x=self.layer[self.num_layers-2](x,frequencies[...,start:end],phase_shifts[...,start:end])
        occupancy=self.to_occupancy[-1](x,occupancy)
        ### 마지막 skip부분, RGB는 제일 마지막에서 accumulate한번 더한다. ray_direction도 같이 들어감.
        start=start+self.hidden_dim
        end=end+self.hidden_dim
        x=self.layer[self.num_layers-1](torch.cat([ray_directions,x],dim=-1),frequencies[...,start:end],frequencies[...,start:end])
        rgb=self.to_RGB[-1](x,rgb)
        
        return torch.cat([rgb,occupancy],dim=-1)
                
        

class TALLSIREN(nn.Module):
    """Primary Generator for GRAM."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ]) 
        self.final_layer = nn.Linear(hidden_dim, 1) # density

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim) # color with direction
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid()) # final color

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) # mapping network!
        # network initialization
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z) # (9,256),(9,256)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30 # frequency magnitude changing

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end]) # filmsiren -> x

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)

