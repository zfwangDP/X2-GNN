# Inputs: structrue of a molecule(str), basis set( set to sto3g), edge_index
# OutPuts: edge_attr, including a flattened hcore matrix & ovlp_matrix, i.e : |edge| x 18

import numpy as np
import pyscf
import torch
from pyscf import gto, dft, lib
from torch_scatter import scatter_add

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
    if type(geom)==str:
        atoms = "\n".join(geom.strip().split("\n")[2:]) 
    else:
        atoms = geom
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
        elif ij_ovlp.size() == torch.Size([9,39]):    # FORGET TO ADD THIS... , # !!!!forgot to add quote.....
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

def geom_scf_grad_6_S_int(geom):
    mol = gto.Mole()
    mol.symmetry = False
    mol.basis = '6-311+G(3df,2p)'
    if type(geom)==str:
        atoms = "\n".join(geom.strip().split("\n")[2:]) 
    else:
        atoms = geom
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
    mat_ovlp = info.get_ovlp()
    # the order of orbitals in aoslices are the same as the matrices, the order of atoms in aoslices are the same as inputs
    aoslice = mol.aoslice_by_atom()     # aoslice: n x 4; first 2 cols are slices of shell(1s, 2s, 2p, etc.), the other 2 cols areslices of orbitals(1s, 2s, 2px, 2py, 2pz, etc.)

    int1e_ipovlp = mol.intor("int1e_ipovlp")
    return mat_ovlp, aoslice, int1e_ipovlp

def fast_gen_edge_grad_6_(mat_ovlp, aoslice, edge_index, int1e_ipovlp):
    int1e_ipovlp = -torch.from_numpy(int1e_ipovlp/lib.param.BOHR)
    mat_ovlp = torch.tensor(mat_ovlp)
    edge_pairs = edge_index.T
    edge_attr = []
    edge_attr_grad = []
    for pair in edge_pairs:
        atom_i, atom_j = pair
        ao_i = aoslice[atom_i][2:]
        ao_j = aoslice[atom_j][2:]

        # slice out the ao_i x ao_j part from the matrices
        ij_ovlp = mat_ovlp[ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]  # 可能是39x39或者9x39 or 39x9的array
        ij_ovlp_grad = int1e_ipovlp[:,ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]
        
        # fill zeros to build uniformly_sized matrices
        ij_ovlp_pad = torch.zeros(39,39)
        ij_ovlp_grad_pad = torch.zeros(3,39,39)
        if ij_ovlp.size() == torch.Size([39,9]):
            ij_ovlp_pad[:,2:11] = ij_ovlp.clone().detach()
            ij_ovlp_grad_pad[:,:,2:11] = ij_ovlp_grad
        elif ij_ovlp.size() == torch.Size([9,9]):
            ij_ovlp_pad[2:11,2:11] = ij_ovlp.clone().detach()
            ij_ovlp_grad_pad[:,2:11,2:11] = ij_ovlp_grad
        elif ij_ovlp.size() == torch.Size([9,39]):
            ij_ovlp_pad[2:11,:] = ij_ovlp.clone().detach()
            ij_ovlp_grad_pad[:,2:11,:] = ij_ovlp_grad
        else:
            ij_ovlp_pad[:ij_ovlp.size()[0],:ij_ovlp.size()[1]] = ij_ovlp.clone().detach()
            ij_ovlp_grad_pad[:,:ij_ovlp.size()[0],:ij_ovlp.size()[1]] = ij_ovlp_grad

        ij_ovlp_pad.requires_grad_(True)

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
        ijSA_ij_ovlp.backward(gradient= torch.ones_like(ijSA_ij_ovlp), inputs = [ij_ovlp_pad])
        ij_ovlp_g_pad = ij_ovlp_grad_pad * ij_ovlp_pad.grad # 39 x 39
        with torch.no_grad():
            jSA_ij_ovlp_grad = torch.cat((ij_ovlp_g_pad[:,:,:5],
                                  ij_ovlp_g_pad[:,:,5:8].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,8:11].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,11:14].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,14:17].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,17:22].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,22:27].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,27:32].sum(dim=2,keepdim=True),
                                  ij_ovlp_g_pad[:,:,32:39].sum(dim=2,keepdim=True),
                                  ), dim = 2)   # SA: symmetry adapted
            ijSA_ij_ovlp_grad = torch.cat((jSA_ij_ovlp_grad[:,:5,:],
                                  jSA_ij_ovlp_grad[:,5:8,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,8:11,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,11:14,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,14:17,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,17:22,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,22:27,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,27:32,:].sum(dim=1,keepdim=True),
                                  jSA_ij_ovlp_grad[:,32:39,:].sum(dim=1,keepdim=True),
                                  ), dim = 1)   # SA: symmetry adapted
            ij_feature = ijSA_ij_ovlp.view(-1)
            ij_feature_grad = ijSA_ij_ovlp_grad.view(3, -1)
            edge_attr.append(ij_feature)
            edge_attr_grad.append(ij_feature_grad)
    edge_attr = torch.stack(edge_attr, dim = 0)
    edge_attr_grad = torch.stack(edge_attr_grad, dim = 1)

    return edge_attr.detach(), edge_attr_grad.detach()

def geom_scf_6_grad_full_(geom):
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
    nao, natm = mol.nao, mol.natm

    int1e_ipovlp = mol.intor("int1e_ipovlp")
    S_1_ao = np.zeros((natm, 3, nao, nao))
    for A in range(natm):
        sA = slice(aoslice[A][2], aoslice[A][3])
        S_1_ao[A, :, sA, :] = - int1e_ipovlp[:, sA, :]
    S_1_ao += S_1_ao.swapaxes(-1, -2)   # from single AO to squared AO

    int1e_ipkin = mol.intor("int1e_ipkin")
    int1e_ipnuc = mol.intor("int1e_ipnuc")
    Z_A = mol.atom_charges()
    H_1_ao = np.zeros((natm, 3, nao, nao))
    for A in range(natm):
        sA = slice(aoslice[A][2], aoslice[A][3])
        H_1_ao[A, :, sA, :] -= int1e_ipkin[:, sA, :]
        H_1_ao[A, :, sA, :] -= int1e_ipnuc[:, sA, :]
        with mol.with_rinv_as_nucleus(A):
            H_1_ao[A] -= Z_A[A] * mol.intor("int1e_iprinv")
    H_1_ao += H_1_ao.swapaxes(-1, -2)    

    return mat_ovlp, mat_hcore/nelectrons, aoslice, S_1_ao, H_1_ao/nelectrons

def gen_edge_grad_6_full_(mat_ovlp, mat_hcore, aoslice, edge_index, ovlp_grad, hcore_grad):
    ovlp_grad = torch.from_numpy(ovlp_grad/lib.param.BOHR)
    hcore_grad = torch.from_numpy(hcore_grad/lib.param.BOHR)
    natm = hcore_grad.size()[0]
    edge_pairs = edge_index.T
    edge_attr = []
    edge_attr_grad = []
    for pair in edge_pairs:
        atom_i, atom_j = pair   # 原子编号，可以用来索引原子序数
        ao_i = aoslice[atom_i][2:]
        ao_j = aoslice[atom_j][2:]
        # slice out the ao_i x ao_j part from the matrices
        ij_ovlp = mat_ovlp[ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]  # 可能是39x39或者9x39或者39×9的array
        ij_hcore = mat_hcore[ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]   #同理
        ij_ovlp_grad = ovlp_grad[:,:,ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]
        ij_hcore_grad = hcore_grad[:,:,ao_i[0]:ao_i[1],ao_j[0]:ao_j[1]]

        ij_ovlp_pad = torch.zeros(39,39)
        ij_hcore_pad = torch.zeros(39,39)
        ij_ovlp_g_pad = torch.zeros(natm,hcore_grad.size()[1],39,39)
        ij_hcore_g_pad = torch.zeros(natm,hcore_grad.size()[1],39,39)
        if ij_ovlp.size() == torch.Size([39,9]):
            ij_hcore_pad[:,2:11] = ij_hcore
            ij_ovlp_pad[:,2:11] = ij_ovlp
            ij_ovlp_g_pad[:,:,:,2:11] = ij_ovlp_grad
            ij_hcore_g_pad[:,:,:,2:11] = ij_hcore_grad
        elif ij_ovlp.size() == torch.Size([9,9]):
            ij_hcore_pad[2:11,2:11] = ij_hcore
            ij_ovlp_pad[2:11,2:11] = ij_ovlp
            ij_ovlp_g_pad[:,:,2:11,2:11] = ij_ovlp_grad
            ij_hcore_g_pad[:,:,2:11,2:11] = ij_hcore_grad
        elif ij_ovlp.size() == torch.Size([9,39]):    # FORGET TO ADD THIS...
            ij_hcore_pad[2:11,:] = ij_hcore
            ij_ovlp_pad[2:11,:] = ij_ovlp
            ij_ovlp_g_pad[:,:,2:11,:] = ij_ovlp_grad
            ij_hcore_g_pad[:,:,2:11,:] = ij_hcore_grad
        else:
            ij_hcore_pad[:ij_hcore.size()[0],:ij_hcore.size()[1]] = ij_hcore
            ij_ovlp_pad[:ij_ovlp.size()[0],:ij_ovlp.size()[1]] = ij_ovlp
            ij_ovlp_g_pad[:,:,:,:] = ij_ovlp_grad
            ij_hcore_g_pad[:,:,:,:] = ij_hcore_grad

        ij_ovlp_pad.requires_grad_(True)
        ij_hcore_pad.requires_grad_(True)

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

        ijSA_ij_ovlp.backward(gradient= torch.ones_like(ijSA_ij_ovlp), inputs = [ij_ovlp_pad])
        ij_ovlp_g_pad = ij_ovlp_g_pad * ij_ovlp_pad.grad # 39 x 39
        ijSA_ij_hcore.backward(gradient= torch.ones_like(ijSA_ij_hcore), inputs = [ij_hcore_pad])
        ij_hcore_g_pad = ij_hcore_g_pad * ij_hcore_pad.grad

        grad_tensor_index = torch.tensor([0,1,2,3,4] + [5]*3 + [6]*3 + [7]*3 + [8]*3 + [9]*5 + [10]*5 + [11]*5 + [12]*7)
        with torch.no_grad():
            jSA_ij_ovlp_grad =  scatter_add(src=ij_ovlp_g_pad, index=grad_tensor_index, dim=3)  # SA: symmetry adapted
            ijSA_ij_ovlp_grad = scatter_add(src=jSA_ij_ovlp_grad, index=grad_tensor_index, dim=2)
            jSA_ij_hcore_grad =  scatter_add(src=ij_hcore_g_pad, index=grad_tensor_index, dim=3)  # SA: symmetry adapted
            ijSA_ij_hcore_grad = scatter_add(src=jSA_ij_hcore_grad, index=grad_tensor_index, dim=2)
            ij_feature_grad = torch.cat((ijSA_ij_ovlp_grad.view(natm, 3, -1), ijSA_ij_hcore_grad.view(natm, 3, -1)), dim=-1)    # |natm| x |3| x |338|
            edge_attr_grad.append(ij_feature_grad)
    edge_attr = torch.stack(edge_attr, dim = 0)
    edge_attr_grad = torch.stack(edge_attr_grad, dim = 2)

    return edge_attr.detach(), edge_attr_grad.detach()  #总之形状需要能对上