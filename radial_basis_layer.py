import numpy as np
import torch
import torch.nn as nn
from envelop import poly_envelop
import math

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
            self.frequencies = np.pi * torch.arange(1, embedding_size+1, dtype = torch.float32, requires_grad = False)
    
    def forward(self, bond_distances):  # input: 1 x n vector
        d_scaled = bond_distances * self.inv_cutoff # 1xn
        d_scaled = d_scaled.unsqueeze(-1)

        return torch.sin(self.frequencies * d_scaled)

class BesselBasis(nn.Module):
    def __init__(self,rbf_dim,cutoff,Trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.inv_cutoff = 1/cutoff
        if Trainable:
            self.frequencies = nn.Parameter(np.pi * torch.arange(1, rbf_dim+1, dtype = torch.float32, requires_grad = True))
        else:
            self.frequencies = np.pi * torch.arange(1, rbf_dim+1, dtype = torch.float32, requires_grad = False)
        
    def forward(self,distances):
        distances = distances.unsqueeze(-1)
        d_scaled = distances * self.inv_cutoff

        return math.sqrt(2*self.inv_cutoff)*torch.sin(self.frequencies * d_scaled)*(distances**(-1))

class GaussianSmearing(nn.Module):
    def __init__(self, num_basis: int, cutoff: float, eps=1e-8):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.eps = eps
        self.mean = torch.nn.Parameter(torch.empty((1, num_basis)))
        self.std = torch.nn.Parameter(torch.empty((1, num_basis)))
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.uniform_(self.mean, 0, self.cutoff)
        torch.nn.init.uniform_(self.std, 1.0 / self.num_basis, 1)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [nedge, 1]
        std = self.std.abs() + self.eps
        coeff = 1 / (std * math.sqrt(2 * math.pi))
        rbf = coeff * torch.exp(-0.5 * ((dist - self.mean) / std) ** 2)
        return rbf
