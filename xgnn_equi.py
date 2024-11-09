import torch
import torch.nn as nn
from e3nn.o3 import SphericalHarmonics,Irreps
from initializer import Glorot_Ortho_
from edge_graph import vertex_to_edge
from envelop import poly_envelop, poly_envelop_std
from atom_embedding import EmbeddingBlock
from torch_geometric.data import Data
from radial_basis_layer import RadialBasis,BesselBasis
from torch_scatter import scatter_add
from initializer import Glorot_Ortho_
from equi_model import SE3Transformer, SE3TransformerV2, xpainn, SE3TransformerV1_5

class DataPre(nn.Module):
    def __init__(self, conv_layers = 3, rbf_dim=20, vector_irreps = "32x1e+16x2e", heads = 8, hidden_dim = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = nn.SiLU()
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SE3Transformer(conv_layers=conv_layers, rbf_dim=rbf_dim, in_channels = hidden_dim, vector_irreps=vector_irreps, heads = heads)
        self.vector_irreps = Irreps(vector_irreps)
        self.rsh_layer = SphericalHarmonics(vector_irreps, normalize=True, normalization="norm")
        
        self.mat_trans = torch.nn.Linear(338, hidden_dim)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, hidden_dim)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(hidden_dim, hidden_dim)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)
    def forward(self, data):
        bond_vector = data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]
        bond_distances = torch.norm(bond_vector, dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        # calculate sphrical expansion
        rsh = self.rsh_layer(bond_vector)
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x_scalar = neo_x, x_vector = rsh, edge_index = neo_edge_index, batch = batch, node_rbf = node_rbf, rsh = rsh)
        return neo_data,data.edge_index[0], data.batch, envelop_para

class xPaiNN(nn.Module):
    def __init__(self, conv_layers = 3, rbf_dim=20, vector_irreps = "128x0e+64x1e+32x2e", hidden_dim = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = nn.SiLU()
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = xpainn(conv_layers=conv_layers, rbf_dim=rbf_dim, vector_irreps=vector_irreps, in_channels=hidden_dim)
        self.vector_irreps = Irreps(vector_irreps)
        self.rsh_layer = SphericalHarmonics(vector_irreps, normalize=True, normalization="norm")
        
        self.mat_trans = torch.nn.Linear(338, hidden_dim)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.emb_trans = torch.nn.Linear(hidden_dim, hidden_dim)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)    

    def forward(self, data):
        bond_vector = data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]
        bond_distances = torch.norm(bond_vector, dim = 1)
        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]
        rsh = self.rsh_layer(bond_vector)
        node_rbf = self.rbf_layer(bond_distances)
        data.rsh = rsh
        data.rbf = node_rbf
        data.envelop_para = envelop_para
        
        return self.fin_model(data)

class XGNN_Equi(nn.Module):
    def __init__(self, conv_layers = 3, rbf_dim=20, vector_irreps = "32x1e+16x2e", heads = 8, hidden_dim = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = nn.SiLU()
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SE3Transformer(conv_layers=conv_layers, rbf_dim=rbf_dim, in_channels = hidden_dim, vector_irreps=vector_irreps, heads = heads)
        #SE3TransformerV2(conv_layers=conv_layers, rbf_dim=rbf_dim, in_channels = hidden_dim, vector_irreps=vector_irreps, heads = heads)
        self.vector_irreps = Irreps(vector_irreps)
        self.rsh_layer = SphericalHarmonics(vector_irreps, normalize=True, normalization="norm")
        
        self.mat_trans = torch.nn.Linear(338, hidden_dim)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.emb_trans = torch.nn.Linear(hidden_dim, hidden_dim)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)
    
    def forward(self, data):
        bond_vector = data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]
        bond_distances = torch.norm(bond_vector, dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])  # crs not supported by cuda
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        # calculate sphrical expansion
        rsh = self.rsh_layer(bond_vector)
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x_scalar = neo_x, x_vector = rsh, edge_index = neo_edge_index, batch = batch, node_rbf = node_rbf, rsh = rsh)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch, envelop_para=envelop_para)
        
        return results
    
class XGNN_Equi_force_ckpt(nn.Module):
    def __init__(self, conv_layers = 3, rbf_dim=20, vector_irreps = "64x1e+32x2e", heads = 8, hidden_dim = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = nn.SiLU()
        self.envelop_function = poly_envelop_std( cutoff = 5.0, exponent=5)
        self.rbf_layer = BesselBasis(cutoff=5.0, rbf_dim=rbf_dim)
        self.fin_model = SE3TransformerV2(conv_layers=conv_layers, rbf_dim=rbf_dim, in_channels = hidden_dim, vector_irreps=vector_irreps, heads = heads)
        self.vector_irreps = Irreps(vector_irreps)
        self.rsh_layer = SphericalHarmonics(vector_irreps, normalize=True, normalization="norm")
        
        self.mat_trans = torch.nn.Linear(169, hidden_dim)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.emb_trans = torch.nn.Linear(hidden_dim, hidden_dim)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)
    
    def forward(self, data):
        data.atom_pos = data.atom_pos.float()
        data.atom_pos.requires_grad_(True)
        data.edge_attr.requires_grad_(True)

        bond_vector = data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]
        bond_distances = torch.norm(bond_vector, dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])  # crs not supported by cuda
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        # calculate sphrical expansion
        rsh = self.rsh_layer(bond_vector)
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf
        neo_x = self.emb_trans(neo_x)

        neo_data = Data(x_scalar = neo_x, x_vector = rsh, edge_index = neo_edge_index, batch = batch, node_rbf = node_rbf, rsh = rsh)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch, envelop_para=envelop_para)
        
        result = results.sum()
        e_a_g, a_p_g = torch.autograd.grad(outputs = result, inputs = (data.edge_attr, data.atom_pos), create_graph= True)
        data.edge_attr.requires_grad_(False)
        data.edge_attr.requires_grad_(False)

        edge_grads = (data.edge_attr_grad.permute(1,0,2) * e_a_g).sum(dim = [2], keepdim=False) #3x|e|

        S_grads_0 = scatter_add(src = edge_grads.permute(1,0), index = data.edge_index[0], dim = 0, dim_size = data.x.size()[0])
        S_grads_1 = scatter_add(src = -edge_grads.permute(1,0), index = data.edge_index[1], dim = 0, dim_size = data.x.size()[0])
        S_grads = S_grads_0 + S_grads_1

        force = -(a_p_g + S_grads)

        return results, force
    
class XGNN_Equi_force(nn.Module):
    def __init__(self, conv_layers = 4, rbf_dim=20, vector_irreps = "32x1e+32x2e", heads = 16, hidden_dim = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = nn.SiLU()
        self.envelop_function = poly_envelop_std( cutoff = 5.0, exponent=5)
        self.rbf_layer = BesselBasis(cutoff=5.0, rbf_dim=rbf_dim)
        self.fin_model = SE3Transformer(conv_layers=conv_layers, rbf_dim=rbf_dim, in_channels = hidden_dim, vector_irreps=vector_irreps, heads = heads)
        self.vector_irreps = Irreps(vector_irreps)
        self.rsh_layer = SphericalHarmonics(vector_irreps, normalize=True, normalization="norm")
        
        self.mat_trans = torch.nn.Linear(169, hidden_dim)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.emb_trans = torch.nn.Linear(hidden_dim, hidden_dim)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)
    
    def forward(self, data):
        data.atom_pos = data.atom_pos.float()
        data.atom_pos.requires_grad_(True)
        data.edge_attr.requires_grad_(True)

        bond_vector = data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]
        bond_distances = torch.norm(bond_vector, dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])  # crs not supported by cuda
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        # calculate sphrical expansion
        rsh = self.rsh_layer(bond_vector)
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf
        neo_x = self.emb_trans(neo_x)

        neo_data = Data(x_scalar = neo_x, x_vector = rsh, edge_index = neo_edge_index, batch = batch, node_rbf = node_rbf, rsh = rsh)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch, envelop_para=envelop_para)
        
        result = results.sum()
        e_a_g, a_p_g = torch.autograd.grad(outputs = result, inputs = (data.edge_attr, data.atom_pos), create_graph= True)
        data.edge_attr.requires_grad_(False)
        data.edge_attr.requires_grad_(False)

        edge_grads = (data.edge_attr_grad.permute(1,0,2) * e_a_g).sum(dim = [2], keepdim=False) #3x|e|

        S_grads_0 = scatter_add(src = edge_grads.permute(1,0), index = data.edge_index[0], dim = 0, dim_size = data.x.size()[0])
        S_grads_1 = scatter_add(src = -edge_grads.permute(1,0), index = data.edge_index[1], dim = 0, dim_size = data.x.size()[0])
        S_grads = S_grads_0 + S_grads_1

        force = -(a_p_g + S_grads)

        return results, force
    
class XGNN_Equi_force_EMB(nn.Module):
    def __init__(self, conv_layers = 3, rbf_dim=20, vector_irreps = "64x1e+32x2e", heads = 8, hidden_dim = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = nn.SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=hidden_dim)
        self.envelop_function = poly_envelop_std( cutoff = 5.0, exponent=5)
        self.rbf_layer = BesselBasis(cutoff=5.0, rbf_dim=rbf_dim)
        self.fin_model = SE3Transformer(conv_layers=conv_layers, rbf_dim=rbf_dim, in_channels = hidden_dim, vector_irreps=vector_irreps, heads = heads)
        self.vector_irreps = Irreps(vector_irreps)
        self.rsh_layer = SphericalHarmonics(vector_irreps, normalize=True, normalization="norm")

        self.rbf_trans = torch.nn.Linear(rbf_dim, hidden_dim)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.mat_trans = torch.nn.Linear(hidden_dim, hidden_dim)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.emb_trans = torch.nn.Linear(2*hidden_dim, hidden_dim)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)
    
    def forward(self, data):
        atom_embeddings = self.emb_block(data.x)
        data.atom_pos = data.atom_pos.float()
        data.atom_pos.requires_grad_(True)

        data.edge_attr = (atom_embeddings[data.edge_index[0]]+atom_embeddings[data.edge_index[1]])/2
        bond_vector = data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]
        bond_distances = torch.norm(bond_vector, dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(data.x.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(data.x.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(data.x.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])  # crs not supported by cuda
        neo_edge_index = neo_edge_index.to(data.x.device)
        neo_x = self.AF(self.mat_trans(data.edge_attr))
        neo_x = torch.cat((neo_x, self.AF(self.rbf_trans(node_rbf*envelop_para))), dim=1)
        neo_x = self.emb_trans(neo_x)
        

        # calculate sphrical expansion
        rsh = self.rsh_layer(bond_vector)
        
        neo_data = Data(x_scalar = neo_x, x_vector = rsh, edge_index = neo_edge_index, batch = batch, node_rbf = node_rbf, rsh = rsh)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch, envelop_para=envelop_para)
        
        result = results.sum()
        a_p_g = torch.autograd.grad(outputs = result, inputs = (data.atom_pos), create_graph= True)[0]

        return results, -a_p_g
    