import torch
from torch.nn import Linear, Sequential, SiLU, ModuleList
from readout import AtomWise, MolWise
from torch_geometric.nn import LayerNorm
from sbftransformer_conv import SBFTransformerConv
import torch.nn.functional as F
from residual_layer import ResidualLayer
from initializer import he_orthogonal_init, Glorot_Ortho_
from torch_scatter import scatter_add

class SBFTransformer(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(data.batch.max()) + 1)
        return results.view(-1)

class SBFTransformerGlobal(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8, pool_option='mean'):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        self.readouts = ModuleList([MolWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1, pool_option=pool_option) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0], atom_batch = atom_batch, dim_size = int(data.batch.max()) + 1)

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0], atom_batch = atom_batch, dim_size = int(data.batch.max()) + 1)

        return results.view(-1)

class SBFTransformerV2(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, K = 2, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = ModuleList([Sequential(Linear(in_channels,in_channels), SiLU(), Linear(in_channels,in_channels)) for i in range(conv_layers)])
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        for conv in self.convs:
            Glorot_Ortho_(conv.lin_sbf.weight)
            Glorot_Ortho_(conv.lin_rbf.weight)
            torch.nn.init.zeros_(conv.lin_sbf.bias)
        for edge_nn in self.edgenn:
            Glorot_Ortho_(edge_nn[0].weight)
            torch.nn.init.zeros_(edge_nn[0].bias)
            Glorot_Ortho_(edge_nn[2].weight)
            torch.nn.init.zeros_(edge_nn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        out = data.x
        results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        for i in range(self.conv_layers):
            out_res_0 = out
            atoms_rep = scatter_add(out, dim=0, index = edge_index_0, dim_size = atom_batch.size()[0])
            edge_attr = atoms_rep[data.edge_attr]
            edge_attr = self.edgenn[i](edge_attr)
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(data.batch.max()) + 1)
        return results.view(-1)/self.conv_layers
    
