import torch.nn as nn
from initializer import he_orthogonal_init, Glorot_Ortho_
from torch_geometric.nn import LayerNorm

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, bias = True):
        super().__init__()
        self.lin0 = nn.Linear(in_channels, in_channels, bias = bias)
        self.lin1 = nn.Linear(in_channels, in_channels, bias = bias)
        self.AF = nn.SiLU()
        #self.Norm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)

        self.reset_parameters()
    
    def reset_parameters(self):
        Glorot_Ortho_(self.lin0.weight)
        nn.init.zeros_(self.lin0.bias)
        Glorot_Ortho_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)

    def forward(self, x):
        out = self.lin0(x)
        out = self.AF(out)
        out = self.lin1(out)
        out = self.AF(out)

        return out+x