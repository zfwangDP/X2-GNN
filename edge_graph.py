import numpy as np
import torch
from atom_embedding import EmbeddingBlock
from torch_geometric.data import Data
from atom_graph import calculate_Dij
import scipy.sparse as sp

# Inputs: edge_index & num_atoms
# Outputs: edge_index in edge_graph and triplets atoms
# deals with any directed graph.(or bidirected)

def vertex_to_edge_2(edge_index, num_nodes):    
    edge_id = torch.arange(edge_index[0].size()[0])
    adj_matrix = sp.coo_matrix((torch.ones(edge_index[0].size()[0]), edge_index), (num_nodes,num_nodes)).tocsr()
    nangles = torch.from_numpy(adj_matrix[edge_index[1]].sum(1).A.T).squeeze(0)
    edge_i = np.repeat(edge_index[0], nangles)  # starting atom
    edge_j = np.repeat(edge_index[1], nangles)  # media atom

    edge_k = torch.from_numpy(adj_matrix[edge_index[1]].nonzero()[1]).long()  # ending atom    # rate determining step2
    angle_res = (edge_i != edge_k).nonzero().T.squeeze()
    edge_id_matrix = sp.coo_matrix((edge_id, edge_index),(num_nodes,num_nodes)).tocsr()

    edge_i = edge_i[angle_res]
    edge_j = edge_j[angle_res]
    edge_k = edge_k[angle_res]
    ij_idx = torch.from_numpy(edge_id_matrix[edge_index[1], :].tocoo().row[angle_res])
    jk_idx = torch.from_numpy(edge_id_matrix[edge_index[1],:].data[angle_res])
    triplets_index = torch.stack([jk_idx,ij_idx])

    return triplets_index, edge_j, edge_i, edge_k
