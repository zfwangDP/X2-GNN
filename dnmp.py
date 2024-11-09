import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear, SiLU, Sequential
from initializer import Glorot_Ortho_

class DNMP(MessagePassing):
    def __init__(self, in_channels, int_emb_size, rbf_dim, sbf_dim, emb_size):
        super().__init__(aggr = 'add')
        self.AF = SiLU()
        self.DP = Linear(emb_size, int_emb_size)
        self.rbf_filter = Linear(rbf_dim, emb_size, bias = False)
        self.sbf_filter = Linear(sbf_dim*rbf_dim, int_emb_size, bias = True)
        self.UP = Linear(int_emb_size, emb_size)
        self.src_trans = Linear(in_channels, emb_size)
        self.dst_trans = Linear(in_channels, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        Glorot_Ortho_(self.DP.weight)
        torch.nn.init.zeros_(self.DP.bias)
        Glorot_Ortho_(self.UP.weight)
        torch.nn.init.zeros_(self.UP.bias)
        Glorot_Ortho_(self.rbf_filter.weight)
        Glorot_Ortho_(self.sbf_filter.weight)
        torch.nn.init.zeros_(self.sbf_filter.bias)
        Glorot_Ortho_(self.src_trans.weight)
        torch.nn.init.zeros_(self.src_trans.bias)
        Glorot_Ortho_(self.dst_trans.weight)
        torch.nn.init.zeros_(self.dst_trans.bias)

    def forward(self, x, rbf, sbf, edge_index):
        rbf_filter = self.rbf_filter(rbf)
        sbf_filter = self.sbf_filter(sbf)

        x_src = x_dst = x
        x_src = self.AF(self.src_trans(x_src))
        x_dst = self.AF(self.dst_trans(x_dst))
        
        x_src = x_src * rbf_filter
        x_src = self.AF(self.DP(x_src))

        x = (x_src, x_dst)
        out = self.propagate(edge_index, x = x, sbf = sbf_filter)

        return out

    def message(self, x_j, sbf):
        # x_j: x_src[edge_index[0]]
        return x_j * sbf

    def update(self, aggr_out, x):
        # should be x_dst + aggr_out
        # aggr_out = scatter_add(self.message, index = edge_index[1], dim = 0)
        aggr_out = self.AF(self.UP(aggr_out))

        return x[1] + aggr_out