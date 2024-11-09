import os
from pathlib import Path
import torch
import numpy as np
import multiprocessing
from multiprocessing import Pool
from utils import Mol_Object
from scf import geom_scf_grad_6_S_int
from atom_graph import calculate_Dij, gen_bonds_mini
from torch_geometric.data import Data, InMemoryDataset
from utils import fast_gen_edge_grad_6_

import datetime

class Mol_Object:
    def __init__(self, this_R, this_Z, this_N, this_Label, this_idx, this_force = None):
        self.R = torch.tensor(this_R)
        self.Z = torch.tensor(this_Z)
        self.N = torch.tensor(this_N)
        self.Label = torch.tensor(this_Label)
        self.idx = torch.tensor(this_idx, dtype = torch.int64)
        self.force = torch.tensor(this_force)

def rmd17_convert(file_name):
    mol_list = []
    f = np.load('/share/home/zfwang/continuous/bgnn/model/layers/rmd17/npz_data/rmd17_%s.npz'%file_name, allow_pickle = True)
    coords = f['coords']
    energies = f['energies']
    forces = f['forces']
    nc = f['nuclear_charges']
    for i,idx in enumerate(f['old_indices']):
        mol_list.append(Mol_Object(this_R=coords[i], this_Z = nc, this_N=len(nc), this_Label= energies[i],
                                    this_force= forces[i], this_idx = i,))
    return mol_list

def mapping(mol_object):
    mat_ovlp, aoslice, int1e_ipovlp = geom_scf_grad_6_S_int([[a_n, tuple(mol_object.R[i])] for i,a_n in enumerate(mol_object.Z)])
    Dij = calculate_Dij(mol_object.R)
    edge_index = gen_bonds_mini(Dij, cutoff = 5.0)
    edge_attr, edge_attr_grad = fast_gen_edge_grad_6_(mat_ovlp, aoslice, edge_index, int1e_ipovlp)
    
    edge_attr_grad = edge_attr_grad.permute(1,0,2)

    return Data(x = mol_object.Z, edge_index = edge_index, edge_attr = edge_attr.float(),  y = mol_object.Label.float(), edge_num = edge_index.size()[1],  edge_attr_grad = edge_attr_grad.float(), 
                idx = mol_object.idx, atom_pos = mol_object.R.float(), force_label = mol_object.force.float())

def paralle(mol_list):
    #cpus = multiprocessing.cpu_count()
    cpus = 32
    print("using %d cores"%cpus)
    P = Pool(processes=cpus)
    datas = P.map(func=mapping, iterable=mol_list)
    P.close()
    P.join()

    return datas

class R_F_BIMD17_6_grad(InMemoryDataset):
    def __init__(self, root = '.', name = 'uracil', transform = None, pre_transform = None, pre_filter = None):
        self.root = Path(root)
        self.input_file = name
        if self.input_file=="all":
            self.prefix = '_'.join(['uracil', 'paracetamol', 'aspirin', 'azobenzene', 'benzene', 'ethanol', 'naphthalene', 'salicylic', 'toluene', 'malonaldehyde'])
        else:
            self.prefix = self.input_file.split('.')[0]
        if '/' in self.input_file:
            self.prefix = self.input_file.split('/')[-1].split('.')[0]
            self.suffix = self.input_file.split('/')[-1].split('.')[1]
        super(R_F_BIMD17_6_grad, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.input_file

    @property
    def processed_dir(self):
        return '/share/home/zfwang/continuous/bgnn/model/layers/processed'

    @property
    def processed_file_names(self):
        return 'rmd17_bi_' + self.prefix + "_6_grad_fast.pt"
    
    def download(self):
        pass

    def process(self):
        #np.random.seed(2023)
        mol_list = rmd17_convert(self.input_file)
        #permutation = np.random.permutation(len(mol_list))  # select randomly for training(sampling)
        #select_index = permutation[:]
        #select = [mol_list[i] for i in select_index]
        datas = paralle(mol_list)
        
        torch.save(self.collate(datas),self.processed_dir +'/' + 'rmd17_bi_' + self.prefix + "_6_grad_fast.pt")
        print('done')

if __name__=='__main__':
    import datetime
    start_time = datetime.datetime.now()

    torch.multiprocessing.set_sharing_strategy('file_system')

    import sys
    from torch.utils.data import dataloader
    from torch.multiprocessing import reductions
    from multiprocessing.reduction import ForkingPickler
 
    default_collate_func = dataloader.default_collate
  
    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)
 
    setattr(dataloader, 'default_collate', default_collate_override)
 
    for t in torch._storage_classes:
      if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
      else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

    mol_name = "naphthalene"
    dataset = R_F_BIMD17_6_grad(name=mol_name)
    end_time = datetime.datetime.now()
    print(f'time consumed: {-(start_time - end_time).total_seconds() :.2f}')