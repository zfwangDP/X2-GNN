import torch
import torch.nn as nn
import numpy as np
from angular_basis_layer import AngularBasisLayer, F_B_2D
from model import SBFTransformerV3
from edge_graph import vertex_to_edge_2
from torch.nn import SiLU
from atom_embedding import EmbeddingBlock
from envelop import poly_envelop
from torch_geometric.data import Data
from radial_basis_layer import RadialBasis
from torch_scatter import scatter_add
from initializer import Glorot_Ortho_

class xgnn_poly(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256, K = 2, heads = 16, embedding_size = 128, device = 'cuda'):
        super().__init__()
        self.device = device

        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size).to(self.device)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5 ).to(self.device)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5).to(self.device)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim).to(self.device)
        self.fin_model = SBFTransformerV3(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, K = K, heads = heads).to(self.device)
        self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)

        self.reset_parameters()
    
    def reset_parameters(self):
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge_2(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, is_bond = data.is_bond, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        return results
    
