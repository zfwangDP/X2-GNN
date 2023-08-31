import torch
import torch.nn as nn
import torch_geometric
from initializer import Glorot_Ortho_
import numpy as np

# inputs: an indicator tensor <atomic_num> specifys the atom type
# outputs: atom embedding matrix, each row represents an atom

class EmbeddingBlock(nn.Module):
    def __init__(self, embedding_size = 128, activation = True, **kwargs):
        super().__init__(**kwargs)
        self.AF = nn.SiLU()
        self.embedding = nn.Embedding(10, embedding_size, padding_idx=0, max_norm = 3.0, scale_grad_by_freq=True)
        self.lin = nn.Linear(embedding_size, embedding_size, bias = True)
        Glorot_Ortho_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
        self.activate = activation
    
    def forward(self, atomic_num):
        if self.activate:
            return self.AF(self.lin(self.embedding(atomic_num)))
        
        else:
            return self.lin(self.embedding(atomic_num))

class EmbeddingBlock_0(nn.Module):
    def __init__(self, embedding_size = 128, **kwargs):
        super().__init__(**kwargs)
        self.kernel = torch.empty(10,embedding_size).uniform_(-np.sqrt(3), np.sqrt(3))
        self.embedding = torch.nn.Parameter(self.kernel)
    
    def forward(self, atomic_num):
            return self.embedding[atomic_num]