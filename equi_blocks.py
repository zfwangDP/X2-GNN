import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps
from residual_layer import ResidualLayer
from initializer import Glorot_Ortho_
from typing import Optional, Tuple, Union, Iterable
from e3nn.util.jit import compile_mode
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_mean, scatter

#@compile_mode("script")
class Invariant(nn.Module):
    """
    Invariant layer.
    """
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        squared: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            `irreps_in`: Input irreps.
            `squared`: Whether to square the norm.
            `eps`: Epsilon for numerical stability.
        """
        super().__init__()
        self.squared = squared
        self.eps = eps
        self.invariant = o3.Norm(irreps_in, squared=squared)

    def forward(self, x):
        if self.squared:
            x = self.invariant(x)
        else:
            x = self.invariant(x + self.eps ** 2) - self.eps
        return x

#@compile_mode("script")
class EquiLayerNorm(nn.Module): # !!! do not pass invariants in!!!
    def __init__(self,
                 in_irreps: Union[str, o3.Irreps]='32x1e+16x2e',
                 eps: float = 1e-8,
                 affine: bool = True,
                 mode: str = "node",    # implemented as in EquiformerV2(arxiv2306.12059)
                 ):
        super().__init__()
        self.eps = eps
        self.irreps = o3.Irreps(in_irreps)  # simplify it before to prevent incorrect mean
        self.num_invariants = self.irreps.count("0e")+self.irreps.count("0o")
        assert self.num_invariants==0, "do not pass invariants in!!!"
        self.multi = torch.tensor([[mul,ir.l*2+1] for i,(mul,ir) in enumerate(self.irreps)]).T  # 2 x nums_l
        #self.multiplicity = torch.tensor(self.irreps.ls).repeat_interleave(torch.tensor(self.irreps.ls)*2+1) # x.size()[1]
        #self.affiliation = torch.arange(self.irreps.num_irreps).repeat_interleave(torch.tensor(self.irreps.ls))    # x.size()[1]
        
        if affine:
            self.weight = nn.Parameter(torch.empty(self.multi.size()[1]))
            self.reset_parameters()
        else:
            self.register_parameter("weight",None)
        assert mode in ["graph","node"], "unknown mode"
        self.mode = mode

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor]=None, batch_size: Optional[int]=None):
        assert len(x.size())==2, "not implemented dim for this block"
        self.multi = self.multi.to(x.device)

        if self.mode=="graph":
            if batch is not None:
                if batch_size is None:
                    batch_size = int(batch.max()) + 1

                square_sigma_l= scatter_mean(x**2, index=torch.arange(self.multi.size()[1],device=x.device).repeat_interleave(self.multi[0]*self.multi[1]).to(x.device), dim=1)  # |v|x(Lmax+1)
                std = scatter_mean(square_sigma_l, index=batch, dim=0).mean(dim=1,keepdim=True).relu().sqrt()  # |G|x1
                out = x/(std[batch]+self.eps)

                if self.weight is not None:
                    out = out * self.weight.repeat_interleave(self.multi[0]*self.multi[1])

                return out
            else:
                std = scatter_mean(x**2, index=torch.arange(self.multi.size()[1],device=x.device).repeat_interleave(self.multi[0]*self.multi[1]).to(x.device), dim=1).mean().relu().sqrt()   # (Lmax+1)
                out = x/(std+self.eps)
                
                if self.weight is not None:
                    out=out*self.weight.repeat_interleave(self.multi[0]*self.multi[1])
                return out

        else:
            square_sigma_l = scatter_mean(x**2, index=torch.arange(self.multi.size()[1],device=x.device).repeat_interleave(self.multi[0]*self.multi[1]).to(x.device), dim=1) # |v|x (Lmax+1)(assume sorted irps) # averaged by (multiplicity*channel)_ir
            std = square_sigma_l.mean(dim=1,keepdim=True).relu().sqrt()    # |v|x1
            out = x / (std+self.eps)

            if self.weight is not None:
                #print(self.multi.device, self.weight.device, out.device)
                out=out*self.weight.repeat_interleave(self.multi[0]*self.multi[1])

            return out

#@compile_mode("trace")
class EquivariantDot(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        assert features1.shape[-1] == features2.shape[-1] == self.input_dim, \
            "Input tensor must have the same last dimension as the irreps"
        out = self.tp(features1, features2)
        return out

#@compile_mode("script")
class SE3TransformerConvV2(nn.Module):
    def __init__(self,
                in_channels: int = 128,
                in_irreps: Union[Irreps, str] = "64x1e+32x2e",
                heads: int = 8, 
                rbf_dim: int = 20,
                out_dim: int = 16,
                match_dim: int = 32,
                dropout: float = 0.,
                root_res: bool = True,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.match_dim = match_dim
        self.out_dim = out_dim
        self.in_irreps = Irreps(in_irreps)

        self.lin_rbf = nn.Linear(rbf_dim, in_channels, bias=True)
        self.lin_rsh_filter = nn.Linear(in_channels, self.in_irreps.num_irreps)
        self.lin_ireps_filter = nn.Linear(in_channels, self.in_irreps.num_irreps)

        self.lin_key = nn.Linear(in_channels, heads * match_dim)
        self.lin_query = nn.Linear(in_channels, heads * match_dim)
        self.lin_value = nn.Sequential(nn.Linear(in_channels, in_channels),nn.SiLU(),nn.Linear(in_channels, heads*out_dim*3))

        self.mixin_rsh = o3.ElementwiseTensorProduct(self.in_irreps, f"{self.in_irreps.num_irreps}x0e")
        self.mixin_irps = o3.ElementwiseTensorProduct(self.in_irreps, f"{self.in_irreps.num_irreps}x0e")
        self.vector_product = o3.FullyConnectedTensorProduct(irreps_in1=self.in_irreps, irreps_in2=self.in_irreps, irreps_out=self.in_irreps)

        self.dropout=dropout
        self.root_res = root_res
        self.AF = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        Glorot_Ortho_(self.lin_rbf.weight)
        nn.init.zeros_(self.lin_rbf.bias)

        for layer in self.lin_value:
            if isinstance(layer, nn.Linear):
                Glorot_Ortho_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_scalar, x_vector, edge_index, rbf, rsh, envelop_para):
        rbf_filter = self.lin_rbf(rbf) * envelop_para
        x_src = x_scalar * rbf_filter    #似乎做一个升维或者把rbf引入rsh是必要的

        # multi-heads self attention for scalar
        query = self.lin_query(x_src).view(-1,self.heads,self.match_dim)
        key = self.lin_key(x_scalar).view(-1,self.heads,self.match_dim)
        value = self.lin_value(x_src)

        att_coeff = (query[edge_index[1]]*key[edge_index[0]]).sum(dim=-1)/math.sqrt(self.match_dim)
        att_coeff = softmax(att_coeff,edge_index[1],dim=0)
        att_coeff = F.dropout(att_coeff,p=self.dropout, training=self.training)

        # split value into 3 parts
        rsh_filter, ireps_filter, value = torch.split(value,[self.in_channels, self.in_channels, self.in_channels],dim=-1)

        # mix scalar and spherical tensor
        mixed_rsh = self.mixin_rsh(rsh, self.lin_rsh_filter(rsh_filter))
        mixed_ireps = self.mixin_irps(x_vector, self.lin_ireps_filter(ireps_filter))+mixed_rsh

        value = value.view(-1,self.heads,self.out_dim)
        scalar_out = scatter_add((att_coeff.unsqueeze(-1) * value[edge_index[0]]), index = edge_index[1], dim=0).view(-1,self.heads*self.out_dim)
        vector_out = self.vector_product(scatter_add(mixed_ireps[edge_index[0]],index=edge_index[1],dim=0),x_vector)
        #vector_out = scatter_add(self.vector_product(mixed_rsh[edge_index[0]],x_vector[edge_index[1]]), index = edge_index[1], dim=0)

        if self.root_res:
            scalar_out+=x_scalar
            vector_out += x_vector

        return scalar_out, vector_out


class SE3TransformerConv(nn.Module):
    def __init__(self,
                in_channels: int = 128,
                in_irreps: Union[Irreps, str] = "32x1e+16x2e",
                heads: int = 8, 
                rbf_dim: int = 20,
                out_dim: int = 16,
                match_dim: int = 32,
                dropout: float = 0.,
                root_res: bool = True,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.match_dim = match_dim
        self.out_dim = out_dim
        self.in_irreps = Irreps(in_irreps)

        self.lin_rbf = nn.Sequential(nn.Linear(rbf_dim, in_channels, bias=False), nn.Linear(in_channels, in_channels, bias=False))
        self.lin_scalar = nn.Sequential(nn.Linear(in_channels, in_channels, bias=False), nn.Linear(in_channels, self.in_irreps.num_irreps, bias=False))
        self.lin_key = nn.Linear(in_channels, heads * match_dim)
        self.lin_query = nn.Linear(in_channels, heads * match_dim)
        self.lin_value = nn.Sequential(nn.Linear(in_channels, in_channels),nn.SiLU(),nn.Linear(in_channels, heads*out_dim))
        self.AF = nn.SiLU()
        self.mixin = o3.ElementwiseTensorProduct(self.in_irreps, f"{self.in_irreps.num_irreps}x0e")
        self.V_P_instructions = [(i, i, i, "uvv", False, ir.dim) for i, (mul, ir) in enumerate(self.in_irreps)]
        self.vector_product = o3.TensorProduct(irreps_in1=self.in_irreps, irreps_in2=self.in_irreps, irreps_out=self.in_irreps, instructions=self.V_P_instructions)  #

        self.dropout=dropout
        self.root_res = root_res

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lin_rbf:
            if isinstance(layer, nn.Linear):
                Glorot_Ortho_(layer.weight)

    def forward(self, x_scalar, x_vector, edge_index, rbf, rsh, envelop_para):
        mixed_rsh = self.mixin(rsh, self.lin_scalar(x_scalar))
        x_src = x_scalar * envelop_para * self.lin_rbf(rbf)    #似乎做一个升维或者把rbf引入rsh是必要的

        # self attention for 
        query = self.lin_query(x_src).view(-1,self.heads,self.match_dim)
        key = self.lin_key(x_scalar).view(-1,self.heads,self.match_dim)
        value = self.lin_value(x_src).view(-1,self.heads,self.out_dim)

        att_coeff = (query[edge_index[1]]*key[edge_index[0]]).sum(dim=-1)/math.sqrt(self.match_dim)
        att_coeff = softmax(att_coeff,edge_index[1],dim=0)
        att_coeff = F.dropout(att_coeff,p=self.dropout, training=self.training)

        scalar_out = scatter_add((att_coeff.unsqueeze(-1) * value[edge_index[0]]), index = edge_index[1], dim=0).view(-1,self.heads*self.out_dim)
        vector_out = self.vector_product(scatter_add(mixed_rsh[edge_index[0]],index=edge_index[1],dim=0),x_vector)
        #vector_out = scatter_add(self.vector_product(mixed_rsh[edge_index[0]],x_vector[edge_index[1]]), index = edge_index[1], dim=0)

        if self.root_res:
            scalar_out += x_scalar
            vector_out += x_vector

        return scalar_out, vector_out

class PainnMessage(nn.Module):
    """Message function for PaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        num_basis: int = 20,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `num_basis`: Number of the radial basis functions.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim + self.edge_num_irreps * 2
        self.num_basis = num_basis
        # scalar feature
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.scalar_mlp[0].bias)
        nn.init.zeros_(self.scalar_mlp[2].bias)
        # spherical feature
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim)
        nn.init.zeros_(self.rbf_lin.bias)
        # elementwise tensor product
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
        fcut: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `rbf`: Radial basis functions.
            `rsh`: Real spherical harmonics.
            `edge_index`: Edge index.
        Returns:
            `new_scalar`: New scalar features.
            `new_spherical`: New spherical features.
        """        
        scalar_out = self.scalar_mlp(x_scalar)
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        gate_state_spherical, gate_edge_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.edge_num_irreps, self.node_dim],
            dim=-1,
        )
        message_spherical = self.rsh_conv(x_spherical[edge_index[1]], gate_state_spherical)
        edge_spherical = self.rsh_conv(rsh, gate_edge_spherical)
        message_spherical = message_spherical + edge_spherical

        # new_scalar = x_scalar.index_add(0, edge_index[0], message_scalar)
        # new_spherical = x_spherical.index_add(0, edge_index[0], message_spherical)
        new_scalar = x_scalar + scatter(message_scalar, edge_index[0], dim=0)
        new_spherical = x_spherical + scatter(message_spherical, edge_index[0], dim=0)

        return new_scalar, new_spherical

#@compile_mode("script")
class PainnUpdate(nn.Module):
    """Update function for PaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.edge_num_irreps
        # spherical feature
        self.update_U = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.update_V = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.invariant = Invariant(self.edge_irreps)
        self.equidot = EquivariantDot(self.edge_irreps)
        self.dot_lin = nn.Linear(self.edge_num_irreps, self.node_dim, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")
        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.edge_num_irreps, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.update_mlp[0].bias)
        nn.init.zeros_(self.update_mlp[2].bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `new_scalar`: New scalar features.
            `new_spherical`: New spherical features.
        """
        U_spherical = self.update_U(x_spherical)
        V_spherical = self.update_V(x_spherical)

        V_invariant = self.invariant(V_spherical)
        mlp_in = torch.cat([x_scalar, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_vv, a_sv, a_ss = torch.split(
            mlp_out,
            [self.edge_num_irreps, self.node_dim, self.node_dim],
            dim=-1
        )
        d_spherical = self.rsh_conv(U_spherical, a_vv)
        inner_prod = self.equidot(U_spherical, V_spherical)
        inner_prod = self.dot_lin(inner_prod)
        d_scalar = a_sv * inner_prod + a_ss

        return x_scalar + d_scalar, x_spherical + d_spherical

#@compile_mode("script")
class XGNN_e3_update(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.edge_num_irreps
        # spherical feature
        self.update_U = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.update_V = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.invariant = Invariant(self.edge_irreps)
        self.equidot = EquivariantDot(self.edge_irreps)
        self.dot_lin = nn.Linear(self.edge_num_irreps, self.node_dim, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")
        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.edge_num_irreps, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.hidden_dim),
        )

    def reset_parameters(self):
        raise NotImplementedError

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `new_scalar`: New scalar features.
            `new_spherical`: New spherical features.
        """
        U_spherical = self.update_U(x_spherical)
        V_spherical = self.update_V(x_spherical)

        V_invariant = self.invariant(V_spherical)
        mlp_in = torch.cat([x_scalar, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_vv, a_sv, a_ss = torch.split(
            mlp_out,
            [self.edge_num_irreps, self.node_dim, self.node_dim],
            dim=-1
        )
        d_spherical = self.rsh_conv(U_spherical, a_vv)
        inner_prod = self.equidot(U_spherical, V_spherical)
        inner_prod = self.dot_lin(inner_prod)
        d_scalar = a_sv * inner_prod + a_ss

        return x_scalar + d_scalar, x_spherical + d_spherical