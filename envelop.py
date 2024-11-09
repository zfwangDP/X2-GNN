import torch
import math
import numpy as np
import torch.nn as nn

class poly_envelop_std(nn.Module):
    def __init__(self, cutoff, exponent):
        super().__init__()
        self.inv_cutoff = 1/float(cutoff)    # cutoff is expected to be a float
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2


    def forward(self, distances):
        scaled_dis = distances * self.inv_cutoff
        env_val = 1 + self.a * scaled_dis**self.p + self.b * scaled_dis**(self.p+1) + self.c * scaled_dis**(self.p + 2)

        return env_val

class poly_envelop(nn.Module):
    def __init__(self, cutoff, exponent):
        super().__init__()
        self.inv_cutoff = 1/cutoff    # cutoff is expected to be a float
        self.exponent = exponent
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2


    def forward(self, distances):
        scaled_dis = distances * self.inv_cutoff
        env_val = 1 / scaled_dis + self.a * scaled_dis**(self.p - 1) + self.b * scaled_dis**self.p + self.c * scaled_dis**(self.p + 1)

        #return torch.where(scaled_dis<1, env_val, torch.zeros_like(scaled_dis)) # 这个where，似乎是没有必要的
        return env_val

def poly_envelop_func(distances, cutoff = 5.0, exponent = 5):
    inv_cutoff = 1/cutoff
    p = exponent+1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2

    scaled_dis = distances * inv_cutoff
    env_val = 1 / scaled_dis + a * scaled_dis**(p - 1) + b * scaled_dis**p + c * scaled_dis**(p + 1)

    return env_val

class cosine_envelop(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)