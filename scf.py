# Inputs: structrue of a molecule(str), basis set( set to sto3g), edge_index
# OutPuts: edge_attr, including a flattened hcore matrix & ovlp_matrix, i.e : |edge| x 18

import numpy as np
import pyscf
import torch
from pyscf import gto, dft, lib

Atom_spin = [0,
             1,0,
             1,0,1,2,3,2,1,0,
             1,0,1,2,3,2,1,0] # '',H~Ar  / Number of unpaired electrons !!!
Elements = ('', 'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' )

Atoms = {}
basis_set = 'sto3g'
for i in range(1,10):
    Atoms[i] = gto.Mole()
    Atoms[i].atom = Elements[i] + '  0.0  0.0  0.0'
    Atoms[i].spin = Atom_spin[i]
    Atoms[i].basis = basis_set
    Atoms[i].build()

# Inputs: geometry of a molecule(in str)
# Outputs: correspondding matrices: hcore, ovlp, aoslice

def geom_scf_6(geom):
    atoms = "\n".join(geom.strip().split("\n")[2:]) 
    mol = gto.Mole()
    mol.symmetry = False
    mol.basis = '6-311+G(3df,2p)'
    mol.atom = atoms
    try:
        mol.spin = 0
        mol.build()
    except:
        mol.spin = 1
        mol.build()
    info = dft.RKS(mol)
    info.xc = "B3LYP" 
    info.conv_tol = 1e-11
    mat_ovlp = torch.from_numpy(info.get_ovlp())
    mat_hcore = torch.from_numpy(info.get_hcore())
    nelectrons = mol.nelectron
    # the order of orbitals in aoslices are the same as the matrices, the order of atoms in aoslices are the same as inputs
    aoslice = mol.aoslice_by_atom()     # aoslice: n x 4; first 2 cols are slices of shell(1s, 2s, 2p, etc.), the other 2 cols areslices of orbitals(1s, 2s, 2px, 2py, 2pz, etc.)

    return mat_ovlp, mat_hcore/nelectrons, aoslice

def bi_gen_edge_feature_6(mat_ovlp, mat_hcore, aoslice, edge_index, Z):
    edge_pairs = edge_index.T
    edge_attr = []
    for pair in edge_pairs:
        atom_i, atom_j = pair   # 原子编号，可以用来索引原子序数
        ao_i = aoslice[atom_i][2:]
        ao_j = aoslice[atom_j][2:]
        # slice out the ao_i x ao_j part from the matrices
        ij_ovlp = mat_ovlp[ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]  # 可能是39x39或者9x39或者39×9的array
        ij_hcore = mat_hcore[ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]   #同理

        ij_ovlp_pad = torch.zeros(39,39)
        ij_hcore_pad = torch.zeros(39,39)
        if ij_ovlp.size() == torch.Size([39,9]):
            ij_hcore_pad[:,2:11] = ij_hcore
            ij_ovlp_pad[:,2:11] = ij_ovlp     
        elif ij_ovlp.size() == torch.Size([9,9]):
            ij_hcore_pad[2:11,2:11] = ij_hcore
            ij_ovlp_pad[2:11,2:11] = ij_ovlp
        elif ij_ovlp.size == torch.Size([9,39]):    # FORGET TO ADD THIS...
            ij_hcore_pad[2:11,:] = ij_hcore
            ij_ovlp_pad[2:11,:] = ij_ovlp
        else:
            ij_hcore_pad[:ij_hcore.size()[0],:ij_hcore.size()[1]] = ij_hcore
            ij_ovlp_pad[:ij_ovlp.size()[0],:ij_ovlp.size()[1]] = ij_ovlp
        jSA_ij_hcore = torch.cat((ij_hcore_pad[:,:5],
                                  torch.norm(ij_hcore_pad[:,5:8],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,8:11],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,11:14],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,14:17],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,17:22],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,22:27],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,27:32],dim = 1,keepdim = True),
                                  torch.norm(ij_hcore_pad[:,32:39],dim = 1,keepdim = True),
                                  ), dim = 1)   # SA: symmetry adapted
        jSA_ij_ovlp = torch.cat((ij_ovlp_pad[:,:5],
                                  torch.norm(ij_ovlp_pad[:,5:8],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,8:11],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,11:14],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,14:17],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,17:22],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,22:27],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,27:32],dim = 1,keepdim = True),
                                  torch.norm(ij_ovlp_pad[:,32:39],dim = 1,keepdim = True),
                                  ), dim = 1)   # SA: symmetry adapted
        ijSA_ij_ovlp = torch.cat((jSA_ij_ovlp[:5],
                                  torch.norm(jSA_ij_ovlp[5:8],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[8:11],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[11:14],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[14:17],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[17:22],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[22:27],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[27:32],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_ovlp[32:39],dim = 0,keepdim = True),
                                  ), dim = 0)   # SA: symmetry adapted
        ijSA_ij_hcore = torch.cat((jSA_ij_hcore[:5],
                                  torch.norm(jSA_ij_hcore[5:8],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[8:11],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[11:14],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[14:17],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[17:22],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[22:27],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[27:32],dim = 0,keepdim = True),
                                  torch.norm(jSA_ij_hcore[32:39],dim = 0,keepdim = True),
                                  ), dim = 0)   # SA: symmetry adapted
        ij_feature = torch.cat((ijSA_ij_ovlp.view(-1),ijSA_ij_hcore.view(-1)),dim = 0)
        edge_attr.append(ij_feature)
    edge_attr = torch.stack(edge_attr, dim = 0).clone().detach()

    return edge_attr