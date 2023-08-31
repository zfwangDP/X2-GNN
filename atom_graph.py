# this module aims to build an atom Data from a raw sdf file

import numpy as np
import torch
from atom_embedding import EmbeddingBlock
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

#             d H HeLiBeB C N O F
Bond_Type = [[0,0,0,0,0,0,0,0,0,0],     # dummy
             [0,1,0,0,0,0,2,3,4,5],     # H
             [0,0,0,0,0,0,0,0,0,0],     # He
             [0,0,0,0,0,0,0,0,0,0],     # Li
             [0,0,0,0,0,0,0,0,0,0],     # Be
             [0,0,0,0,0,0,0,0,0,0],     # B
             [0,2,0,0,0,0,6,7,8,9],     # C
             [0,3,0,0,0,0,7,10,11,12],  # N
             [0,4,0,0,0,0,8,11,13,14],  # O
             [0,5,0,0,0,0,9,12,14,15]]  # F
Bond_Type = torch.tensor(Bond_Type, requires_grad = False)
B_T = ['none','H-H','H-C','H-N','H-O','H-F','C-C','C-N','C-O','C-F','N-N','N-O','N-F','O-O','O-F','F-F']

# define van der waals radius to figure out the bonds among edges

Vdw_r = torch.tensor([0.0, 1.08, 1.34, 1.75, 2.05, 1.47, 1.49, 1.41, 1.40, 1.39], requires_grad = False)
Vdw_sum = torch.tile(Vdw_r,(10,1)).T + torch.tile(Vdw_r,(10,1))


# inputs: a tensor of atom positions which lines are supposed to represent an atom, size: n x 3
# outputs: a n x n distances matrix, indices same as inputs

def calculate_Dij(atom_pos):
    Gram_matrix = torch.mm(atom_pos, atom_pos.T)
    H = torch.tile(torch.diag(Gram_matrix),(atom_pos.size()[0],1))
    return  torch.relu((H + H.T - 2*Gram_matrix)**0.5)  #use relu to prevent strange numerical problems


# inputs: distances matrix from calculated_Dij, a cutoff distances (float, in Angstorm)
# outputs: edge_index
# output undirected edges(or to say bidirected.)

def gen_bonds_mini(Dij, cutoff = 5.0):
    adj_matrix = (Dij<cutoff) & Dij.bool()
    edge_index = np.argwhere(adj_matrix)
    return edge_index