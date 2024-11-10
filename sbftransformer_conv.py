import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from initializer import Glorot_Ortho_


class SBFTransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        sbf_dim: int = 16,
        rbf_dim: int = 16,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(SBFTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.sbf_dim = sbf_dim
        self.rbf_dim = rbf_dim
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.lin_sbf = Linear(sbf_dim, heads * out_channels, bias = True)
        self.lin_rbf = Linear(rbf_dim, in_channels[0], bias = False)

        self.reset_parameters()

    def reset_parameters(self):
        Glorot_Ortho_(self.lin_sbf.weight)
        Glorot_Ortho_(self.lin_rbf.weight)
        torch.nn.init.zeros_(self.lin_sbf.bias)
        #torch.nn.init.zeros_(self.lin_rbf.bias)

        #Glorot_Ortho_(self.lin_key.weight)
        #Glorot_Ortho_(self.lin_query.weight)
        #Glorot_Ortho_(self.lin_value.weight)
        #if self.edge_dim:
        #    Glorot_Ortho_(self.lin_edge.weight)
        #Glorot_Ortho_(self.lin_skip.weight)
        #if self.beta:
        #    Glorot_Ortho_(self.lin_beta.weight)

    def forward(self, sbf, rbf, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_src = x_dst = x
        rbf_filter = self.lin_rbf(rbf)
        x_src = x_src * rbf_filter

        if isinstance(x, Tensor):
            x: PairTensor = (x_src, x_dst)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, sbf=sbf, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, sbf, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)    # |e| x h x c
            key_j += edge_attr
        
        sbf = self.lin_sbf(sbf)   # |e| x h x 1

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha # |e| x H
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j   # |e| x H x C
        if edge_attr is not None:
            out += edge_attr

        out *= sbf.view(-1, self.heads, self.out_channels)
        out *= alpha.view(-1, self.heads, 1)    # use sbf expension similarly with alpha, #epual to alpha.unsqueeze(-1)
        
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')