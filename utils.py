import torch
from pyscf import lib
import torch.nn as nn
import numpy as np

class Mol_Object:
    def __init__(self, atom, this_R, this_Z, this_N, this_Label, this_idx, this_force = None):
        self.atom = atom
        self.R = torch.tensor(this_R)
        self.Z = torch.tensor(this_Z)
        self.N = torch.tensor(this_N)
        self.Label = torch.tensor(this_Label)
        self.idx = torch.tensor(this_idx, dtype = torch.int64)
        if type(this_force) == list:
            self.force = torch.tensor(this_force)

def read_xyz(filename):
    mol_list = []
    Atom_Number = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9}

    this=''
    this_R = []
    this_N = []
    this_Z = []
    this_y = []
    this_idx = [0]

    with open(filename, 'r')as f:
        lines = f.readlines()
        counter = 0
        end = len(lines)-1
        for i,line in enumerate(lines):
            if line == '\n':    #去掉空行
                continue
            elif len(line.split())==1:
                if type(eval(line)) == int:
                    mol_object = Mol_Object(atom = this, this_R=this_R, this_idx=this_idx, this_Label=this_y, this_Z=this_Z, this_N=this_N)
                    mol_list.append(mol_object)
                    this = ''
                    this_R = []
                    this_N = []
                    this_Z = []
                    this_y = []
                    this_idx = []
                    this_idx.append(counter)
                    counter += 1

                    this += line
                    this_N.append(eval(line))

                elif type(eval(line)) == float:
                    this_y.append(eval(line))
                    this += line

            elif len(line.split()) == 4:
                line = line.split()
                this += (line[0]+' '+line[1]+' '+line[2]+' '+line[3]+'\n')
                this_R.append(list(map(float,line[1:])))
                this_Z.append(Atom_Number[line[0]])
                if i == end:
                    mol_object = Mol_Object(atom = this, this_R=this_R, this_idx=this_idx, this_Label=this_y, this_Z=this_Z, this_N=this_N)
                    mol_list.append(mol_object)   
    return mol_list[1:]
