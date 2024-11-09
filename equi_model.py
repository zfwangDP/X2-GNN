import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, SiLU, ModuleList
from e3nn.o3 import Irreps
from equi_read import AtomWiseInvariants
from equi_blocks import SE3TransformerConv, SE3TransformerConvV2, PainnUpdate, EquiLayerNorm, PainnMessage, XGNN_e3_update
import torch.nn.functional as F
from torch_scatter import scatter_add
from e3nn.util.jit import compile_mode
from torch_geometric.nn import LayerNorm

#@compile_mode("script")
class SE3Transformer(nn.Module):
    def __init__(self, conv_layers=3, rbf_dim=20, vector_irreps="32x1e+16x2e", in_channels=128, heads = 8):
        super().__init__()
        self.vector_irreps = Irreps(vector_irreps)
        self.message = ModuleList([SE3TransformerConv(in_channels, vector_irreps, heads, rbf_dim, int(in_channels/heads), match_dim=32, root_res=True) for i in range(conv_layers)])
        self.update = ModuleList([PainnUpdate(node_dim=in_channels, edge_irreps=self.vector_irreps) for i in range(conv_layers)])
        self.readout = ModuleList([AtomWiseInvariants(mlp_depth = 3, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1) for i in range(conv_layers+1)])
        self.ShpericalNorm = EquiLayerNorm(in_irreps=vector_irreps,affine=True,eps=1e-8,mode="node")
        self.InvariantNorm = LayerNorm(in_channels=in_channels,eps=1e-8,affine=True,mode="node")

    def forward(self, data, edge_index_0, atom_batch, envelop_para):
        x_scalar,x_vector=data.x_scalar, data.x_vector
        x_scalar = self.InvariantNorm(x_scalar)
        results = self.readout[0](x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)

        for msg, upd, rdt in zip(self.message, self.update, self.readout[1:]):
            x_scalar,x_vector = msg(x_scalar, x_vector, data.edge_index, data.node_rbf, data.rsh, envelop_para)
            x_scalar, x_vector = upd(x_scalar, x_vector)
            x_scalar = self.InvariantNorm(x_scalar)
            x_vector = self.ShpericalNorm(x_vector)
            results+=rdt(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0,envelop_para=envelop_para)
        
        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(atom_batch.max()) + 1)
        return results.view(-1)

class SE3TransformerV1_5(nn.Module):  #大的结构不用改
    def __init__(self, conv_layers=3, rbf_dim=20, vector_irreps="64x1e+32x2e", in_channels=128, heads = 8):
        super().__init__()
        assert in_channels%heads==0, r"make sure inchannels%heads==0"
        self.vector_irreps = Irreps(vector_irreps)
        self.message = ModuleList([SE3TransformerConvV2(in_channels, vector_irreps, heads, rbf_dim, int(in_channels/heads), match_dim=32, root_res=True) for i in range(conv_layers)])
        self.update = ModuleList([XGNN_e3_update(node_dim=in_channels, edge_irreps=self.vector_irreps) for i in range(conv_layers)])
        self.readout = ModuleList([AtomWiseInvariants(mlp_depth = 3, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1) for i in range(conv_layers+1)])
        #self.readout = AtomWiseInvariants(mlp_depth = 4, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1)
        self.ShpericalNorm = EquiLayerNorm(in_irreps=vector_irreps,affine=True,eps=1e-8,mode="node")
        self.InvariantNorm = LayerNorm(in_channels=in_channels,eps=1e-8,affine=True,mode="node")

    def forward(self, data, edge_index_0, atom_batch, envelop_para):
        x_scalar,x_vector=data.x_scalar, data.x_vector
        #x_scalar = self.InvariantNorm(x_scalar)
        results = self.readout[0](x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)

        for msg, upd, rdt in zip(self.message, self.update, self.readout[1:]):
        #for msg, upd in zip(self.message, self.update):
            x_scalar_res, x_vector_res = x_scalar, x_vector

            x_scalar,x_vector = msg(x_scalar, x_vector, data.edge_index, data.node_rbf, data.rsh, envelop_para)
            x_scalar, x_vector = upd(x_scalar, x_vector)

            x_scalar = self.InvariantNorm(x_scalar)
            x_vector = self.ShpericalNorm(x_vector)

            x_scalar += x_scalar_res
            x_vector += x_vector_res

            results+=rdt(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0,envelop_para=envelop_para)
        
        #results = self.readout(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)
        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(atom_batch.max()) + 1)
        return results.view(-1)

class SE3TransformerV2(nn.Module):  #大的结构不用改
    def __init__(self, conv_layers=3, rbf_dim=20, vector_irreps="64x1e+32x2e", in_channels=128, heads = 8):
        super().__init__()
        assert in_channels%heads==0, r"make sure inchannels%heads==0"
        self.vector_irreps = Irreps(vector_irreps)
        self.message = ModuleList([SE3TransformerConvV2(in_channels, vector_irreps, heads, rbf_dim, int(in_channels/heads), match_dim=32, root_res=True) for i in range(conv_layers)])
        self.update = ModuleList([XGNN_e3_update(node_dim=in_channels, edge_irreps=self.vector_irreps) for i in range(conv_layers)])
        #self.readout = ModuleList([AtomWiseInvariants(mlp_depth = 3, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1) for i in range(conv_layers+1)])
        self.readout = AtomWiseInvariants(mlp_depth = 4, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1)
        self.ShpericalNorm = EquiLayerNorm(in_irreps=vector_irreps,affine=True,eps=1e-8,mode="node")
        self.InvariantNorm = LayerNorm(in_channels=in_channels,eps=1e-8,affine=True,mode="node")

    def forward(self, data, edge_index_0, atom_batch, envelop_para):
        x_scalar,x_vector=data.x_scalar, data.x_vector
        #x_scalar = self.InvariantNorm(x_scalar)
        #results = self.readout[0](x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)

        #for msg, upd, rdt in zip(self.message, self.update, self.readout[1:]):
        for msg, upd in zip(self.message, self.update):
            x_scalar_res, x_vector_res = x_scalar, x_vector

            x_scalar,x_vector = msg(x_scalar, x_vector, data.edge_index, data.node_rbf, data.rsh, envelop_para)
            x_scalar, x_vector = upd(x_scalar, x_vector)

            x_scalar = self.InvariantNorm(x_scalar)
            x_vector = self.ShpericalNorm(x_vector)

            x_scalar += x_scalar_res
            x_vector += x_vector_res

            #results+=rdt(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0,envelop_para=envelop_para)
        
        results = self.readout(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)
        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(atom_batch.max()) + 1)
        return results.view(-1)

class xpainn(nn.Module):
    def __init__(self, conv_layers=3, rbf_dim=20, vector_irreps="128x0e+64x1e+32x2e", in_channels=128):
        super().__init__()
        self.vector_irreps = Irreps(vector_irreps)
        self.message = ModuleList([PainnMessage(in_channels, vector_irreps, rbf_dim, ) for i in range(conv_layers)])
        self.update = ModuleList([PainnUpdate(node_dim=in_channels, edge_irreps=self.vector_irreps) for i in range(conv_layers)])
        #self.readout = ModuleList([AtomWiseInvariants(mlp_depth = 3, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1) for i in range(conv_layers+1)])
        self.readout = AtomWiseInvariants(mlp_depth = 3, in_channels = in_channels, rbf_dim = rbf_dim, num_target = 1)
        self.ShpericalNorm = EquiLayerNorm(in_irreps=vector_irreps,affine=True,eps=1e-8,mode="node")
        self.InvariantNorm = LayerNorm(in_channels=in_channels,eps=1e-8,affine=True,mode="node")

    def forward(self, data, edge_index_0, atom_batch, envelop_para):
        x_scalar,x_vector=data.x_scalar, data.x_vector
        #x_scalar = self.InvariantNorm(x_scalar)
        #results = self.readout[0](x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)

        for msg, upd, rdt in zip(self.message, self.update, self.readout[1:]):
            x_scalar,x_vector = msg(x_scalar, x_vector, data.edge_index, data.node_rbf, data.rsh, envelop_para)
            x_scalar, x_vector = upd(x_scalar, x_vector)
            x_scalar = self.InvariantNorm(x_scalar)
            x_vector = self.ShpericalNorm(x_vector)
            #results+=rdt(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0,envelop_para=envelop_para)

        results = self.readout(x_scalar=x_scalar, rbf=data.node_rbf, num_atoms = atom_batch.size()[0], edge_index_0 = edge_index_0, envelop_para=envelop_para)
        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(atom_batch.max()) + 1)
        return results.view(-1)