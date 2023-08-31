import torch
import math
from torch.nn import Linear, SiLU, Sequential
import torch.nn.functional as F
from initializer import Glorot_Ortho_
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

class SBFTransformerConv(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                heads: int = 1,
                concat: bool = True,
                beta: bool = False,
                dropout: float = 0.,
                edge_dim: int = None,
                sbf_dim: int = None,
                rbf_dim: int = None,
                bias: bool = True,
                root_weight: bool = True,
                **kwargs,
                ):
        super().__init__()
        self.Heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.beta = beta
        self.bias = bias
        self.root_weight = root_weight
        self.edge_dim = edge_dim
        self.lin_sbf = Sequential(Linear(rbf_dim, in_channels, bias = False),  Linear(in_channels, in_channels, bias = False))
        self.lin_rbf = Sequential(Linear(sbf_dim * 6, in_channels, bias = False),  Linear(in_channels, in_channels, bias = False))
        if edge_dim:
            self.lin_edge_key = Linear(edge_dim, out_channels * heads, bias = False)
            self.lin_edge = Linear(edge_dim, out_channels * heads, bias = False)
        else:
            self.lin_edge_key = self.register_parameter('lin_edge_key', None)
            self.lin_edge = self.register_parameter('lin_edge', None)
        self.lin_key = Linear(in_channels, out_channels * heads)
        self.lin_query = Linear(in_channels, out_channels * heads)
        self.lin_value = Linear(in_channels, out_channels * heads)
        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels,
                                   bias=bias)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)


    def reset_parameters(self):
        Glorot_Ortho_(self.lin_sbf[0].weight)
        Glorot_Ortho_(self.lin_sbf[1].weight)
        Glorot_Ortho_(self.lin_rbf[0].weight)
        Glorot_Ortho_(self.lin_rbf[1].weight)
        if self.edge_dim:
            Glorot_Ortho_(self.lin_edge_key.weight)
            Glorot_Ortho_(self.lin_edge.weight)
        Glorot_Ortho_(self.lin_key.weight)
        torch.nn.init.zeros(self.lin_key.bias)
        Glorot_Ortho_(self.lin_value.weight)
        torch.nn.init.zeros(self.lin_value.bias)
        Glorot_Ortho_(self.lin_query.weight)
        torch.nn.init.zeros(self.lin_query.bias)
        Glorot_Ortho_(self.lin_skip.weight)
        torch.nn.init.zeros(self.lin_skip.bias)        

    def forward(self, x, edge_index, edge_attr, rbf, sbf, edge_index_0):
        H, C = self.Heads, self.out_channels

        rbf_filter = self.lin_rbf(rbf)
        sbf_filter = self.lin_sbf(sbf)
        x_src, x_dst = (x, x)
        x_src = rbf_filter * x_src
        key = self.lin_key(x_src).view(-1, H, C)
        query = self.lin_query(x_dst).view(-1, H, C)
        value = self.lin_value(x_src).view(-1, H, C)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_key = self.lin_edge_key(edge_attr).view(-1, H, C)
            edge_value = self.lin_edge(edge_attr).view(-1, H, C)
        
            key_j = key[edge_index[0]] + edge_key
            value_j = value[edge_index[0]] + edge_value
        
        else:
            key_j = key[edge_index[0]]
            value_j = value[edge_index[0]]

        query_i = query[edge_index[1]]
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, edge_index[1])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        message = value_j * alpha.view(-1, self.heads, 1)
        message = message * sbf_filter

        aggr_out = scatter_add(message, index = edge_index[1], dim = 0)

        if self.concat:
            out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            out = aggr_out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x_dst)
            out = out + x_r
        
        return out
