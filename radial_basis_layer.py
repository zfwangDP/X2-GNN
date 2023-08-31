import numpy as np
import torch
import torch.nn as nn
from envelop import poly_envelop

def radialbasis(r, cutoff, embedding_size): #这里是DimeNet式的rbf
    r"""takes a **single** distance (process data by data) in,
    return an embedding vector
    """
    num = r.shape[0]
    n = torch.arange(embedding_size)+1
    n = n.unsqueeze(0)
    n.repeat(num, 1)
    norm_coefficient = (2/cutoff)**0.5 #share by all
    frequency = r*n*np.pi/cutoff

    return norm_coefficient*torch.sin(frequency)/r

def RadialBasis_func(bond_distances, cutoff=5.0, embedding_size=16):
    inv_cutoff = 1 / cutoff
    frequencies = np.pi * torch.arange(1, embedding_size+1, dtype = torch.float32)
    d_scaled = bond_distances * inv_cutoff
    d_scaled = d_scaled.unsqueeze(-1)
    return torch.sin(frequencies * d_scaled)

class RadialBasis(nn.Module):
    def __init__(self, embedding_size, cutoff, Trainable = True, **kwargs):
        super().__init__(**kwargs)
        self.num_radial = embedding_size
        self.inv_cutoff = 1 / cutoff
        if Trainable:
            self.frequencies = nn.Parameter(np.pi * torch.arange(1, embedding_size+1, dtype = torch.float32, requires_grad = True))
        else:
            self.frequencies = np.pi * torch.arange(1, embedding_size+1, dtype = torch.float32, requires_grad = True)
    
    def forward(self, bond_distances):  # input: 1 x n vector
        d_scaled = bond_distances * self.inv_cutoff # 1xn
        d_scaled = d_scaled.unsqueeze(-1)

        return torch.sin(self.frequencies * d_scaled)