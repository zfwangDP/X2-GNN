import os
from pathlib import Path
import torch
import numpy as np
import multiprocessing
from multiprocessing import Pool
from utils import md17_xyz_read, Mol_Object
from scf import geom_scf_6_grad_full_, gen_edge_grad_6_full_
from atom_graph import calculate_Dij, gen_bonds_mini
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

import datetime

def mapping(mol_object):
    mat_ovlp, mat_hcore, aoslice, ovlp_grad, hcore_grad = geom_scf_6_grad_full_(mol_object.atom)
    Dij = calculate_Dij(mol_object.R)
    edge_index = gen_bonds_mini(Dij,cutoff=5.0)
    edge_attr, edge_attr_grad = gen_edge_grad_6_full_(mat_ovlp, mat_hcore, aoslice, edge_index, ovlp_grad, hcore_grad)
    
    edge_attr_grad = edge_attr_grad.permute(2,0,1,3)

    return Data(x = mol_object.Z, edge_index = edge_index, edge_attr = edge_attr,  y = mol_object.Label, edge_num = edge_index.size()[1],  edge_attr_grad = edge_attr_grad, 
                idx = mol_object.idx, atom_pos = mol_object.R, force_label = mol_object.force)

def paralle(mol_list):
    #cpus = multiprocessing.cpu_count()
    cpus = 32
    print("using %d cores"%cpus)
    P = Pool(processes=cpus)
    datas = P.map(func=mapping, iterable=mol_list)
    P.close()
    P.join()

    return datas

class F_BIMD17_6_Full(InMemoryDataset):
    def __init__(self, root = '.', input_file = './raw/uracil.xyz', length=2000, index_file=None, transform = None, pre_transform = None, pre_filter = None):
        self.root = Path(root)
        self.input_file = input_file
        self.length = length
        self.index_file = index_file
        self.prefix = input_file.split('.')[0]
        self.suffix = input_file.split('.')[1]
        if '/' in self.input_file:
            self.prefix = self.input_file.split('/')[-1].split('.')[0]
            self.suffix = self.input_file.split('/')[-1].split('.')[1]
        super(F_BIMD17_6_Full, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.input_file
    
    @property
    def processed_file_names(self):
        if self.index_file is not None:
            return 'KS_bi_' + self.prefix + "_6_grad_full.pt"
        else:
            return 'bi_' + self.prefix + "_6_grad_full.pt"
    
    @property
    def processed_dir(self):
        return '/share/home/zfwang/XGNN/processed'

    def download(self):
        pass

    def process(self):
        np.random.seed(2023)
        mol_list = md17_xyz_read(self.input_file)
        if self.index_file is None:
            permutation = np.random.permutation(len(mol_list))  # select randomly for training(sampling)
            select_index = permutation[:self.length]
        else:
            select_index = np.load(self.index_file)
        select = tqdm([mol_list[i] for i in select_index])
        datas = paralle(select)
        
        if self.index_file is not None:
            torch.save(self.collate(datas),self.processed_dir +'/' + 'KS_bi_' + self.prefix + "_6_grad_full.pt")
        else:
            torch.save(self.collate(datas),self.processed_dir +'/' + 'bi_' + self.prefix + "_6_grad_full.pt")
        print('done')

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--inp_file',type=str,required=True)
    parser.add_argument('--len',type=int,required=True,default=2000)
    parser.add_argument('--index_file',type=str,required=False,default=None)
    args = parser.parse_args()
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

    mol_name = "uracil"
    dataset = F_BIMD17_6_Full(input_file=args.inp_file, length=args.len, index_file=args.index_file)  # only benzene undo, and ethanol only 1200 points
    end_time = datetime.datetime.now()
    print(f'time consumed: {-(start_time - end_time).total_seconds() :.2f}')