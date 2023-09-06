import torch
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean
from torch.nn import Linear, SiLU, Sequential, ModuleList
from initializer import he_orthogonal_init, Glorot_Ortho_

class AtomWise(torch.nn.Module):
    def __init__(self,
                 mlp_depth = 3,
                 in_channels = 256,
                 rbf_dim = 16,
                 num_target = 1
                 ):
        super().__init__()
        self.mlp = ModuleList(
            [])
        for i in range(mlp_depth-1):
            self.mlp.append(Linear(in_channels,in_channels))
            self.mlp.append(SiLU())
        self.mlp.append(Linear(in_channels,num_target))

        self.lin_rbf = Linear(rbf_dim, in_channels)

        self.reset_parameters()
    
    def reset_parameters(self):
        Glorot_Ortho_(self.lin_rbf.weight)
        torch.nn.init.zeros_(self.lin_rbf.bias)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                Glorot_Ortho_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x , rbf, num_atoms, edge_index_0):
        rbf_filter = self.lin_rbf(rbf)
        out = rbf_filter * x
        out = scatter_add(out, dim=0, index = edge_index_0, dim_size = num_atoms)   # sum edge representations into atom representations

        # map atom representations into atom energy
        for layer in self.mlp:
            out = layer(out)

        return out
    
class MolWise(torch.nn.Module):
    def __init__(self, mlp_depth = 3, in_channels = 256, rbf_dim = 6, num_target = 1, pool_option = 'mean'):
        super().__init__()
        self.lin_rbf = Linear(rbf_dim, in_channels)
        self.mlp = ModuleList([])
        for i in range(mlp_depth-1):
            self.mlp.append(Linear(in_channels,in_channels))
            self.mlp.append(SiLU())
        self.mlp.append(Linear(in_channels, num_target))
        
        assert pool_option in ['mean','add'], 'unsupport pooling option'
        self.pool_option = pool_option

        self.reset_parameters()
    
    def reset_parameters(self):
        Glorot_Ortho_(self.lin_rbf.weight)
        torch.nn.init.zeros_(self.lin_rbf.bias)
    
    def forward(self, x, rbf, num_atoms, edge_index_0, atom_batch, dim_size):
        rbf_filter = self.lin_rbf(rbf)
        out = rbf_filter * x
        out = scatter_add(out, dim=0, index = edge_index_0, dim_size=num_atoms) # sum edge representations into atom representations
        if self.pool_option == 'mean':
            out = scatter_mean(out, index= atom_batch, dim = 0, dim_size=dim_size)
        elif self.pool_option == 'add':
            out = scatter_add(out, index= atom_batch, dim = 0, dim_size=dim_size)
        
        for layer in self.mlp:
            out = layer(out)
        
        return out
