import os
import torch
import torch.nn as nn
import numpy as np
from md17fast_6 import F_BIMD17_6_grad
from md17_full import F_BIMD17_6_Full
from rmd17_6 import R_F_BIMD17_6_grad
from torch_geometric.loader import DataLoader
from trainer import Train_with_force
from xgnn import xgnn_poly_force, xgnn_poly_force_full
from xgnn_equi import XGNN_Equi_force, XGNN_Equi_force_ckpt
from scheduler import LinearWarmupExponentialDecay
from torch_scatter import scatter_add
import json
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

RMD17_Atom_Energy_kcal = torch.tensor([torch.nan, -312.89494, torch.nan, torch.nan, torch.nan, torch.nan,
                                                    -23689.24577, -34178.27749, -47009.64714, -62462.128616 ], device = 'cpu')

args = {}
with open('./config_force.json','rt') as f:
    args.update(json.load(f))

torch.set_num_threads(args['num_thread'])

mol_name = args["mol_name"]
if args["dataset"] == 'md17':
    if args["full_grad"]:
        dataset = F_BIMD17_6_Full(input_file=f'/share/home/zfwang/continuous/bgnn/model/layers/md17/{mol_name}.xyz',length=2000, index_file=f'/share/home/zfwang/KS/md17_{mol_name}_selected.npy')
    else:
        dataset = F_BIMD17_6_grad(names = mol_name)
elif args["dataset"] == 'rmd17':
    dataset = R_F_BIMD17_6_grad(name = mol_name)

dataset.data.y = dataset.data.y * 0.04336414
dataset.data.force_label = dataset.data.force_label * 0.04336414
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)

if args['pre_process']=="remove_mean":
    dataset.data.y = (dataset.data.y - mean)    # using KS sampled data may deviate from the whole dataset
elif args['pre_process']=="remove_atom_ref":
    atom_affi = torch.arange(len(dataset)).repeat_interleave(dataset.slices['x'][1:] - dataset.slices['x'][:-1])
    mol_ref = scatter_add(RMD17_Atom_Energy_kcal[dataset.data.x.long()],index=atom_affi,dim=0)
    dataset.data.y = dataset.data.y.squeeze() - mol_ref

mean, std = mean.item(), std.item()
print("statistics:",mean,std)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args['equi_model']:
    model = XGNN_Equi_force_ckpt(conv_layers = args["conv_layers"], rbf_dim=args["rbf_dim"], vector_irreps = args["vector_irreps"],
                             hidden_dim=args["in_channels"], heads=args["heads"], device=device).to(device)
elif args["full_grad"]:
    if args["S_only"]:
        model = xgnn_poly_force_full(conv_layers = args["conv_layers"], sbf_dim=args['sbf_dim'], rbf_dim=args['rbf_dim'],
                         in_channels = args['in_channels'], K = 2, heads = args['heads'], mat_dim = 169, embedding_size = args['embedding_size'], device = device).to(device)        
    else:
        model = xgnn_poly_force_full(conv_layers = args["conv_layers"], sbf_dim=args['sbf_dim'], rbf_dim=args['rbf_dim'],
                         in_channels = args['in_channels'], K = 2, heads = args['heads'], mat_dim = 338, embedding_size = args['embedding_size'], device = device).to(device)
else:
    model = xgnn_poly_force(conv_layers = args["conv_layers"], sbf_dim=args['sbf_dim'], rbf_dim=args['rbf_dim'],
                         in_channels = args['in_channels'], K = 2, heads = args['heads'], mat_dim = 169, embedding_size = args['embedding_size'], device = device).to(device)

ema_avg = lambda averaged_model_parameter, model_parameter:\
        args["ema_decay"] * averaged_model_parameter + (1-args["ema_decay"]) * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['max_lr'], amsgrad = False)
assert args['scheduler'] in ["LinearWarmupExponentialDecay","ReduceLROnPlateau"], 'unimplemented LR scheduler'
if args['scheduler']=="LinearWarmupExponentialDecay":
    scheduler = LinearWarmupExponentialDecay(optimizer, warmup_steps = args['warmup_steps'],decay_steps=args['decay_steps'], decay_rate = args['decay_rate'])
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args['reduce_factor'], patience=args['patience'], min_lr=args['max_lr']*args['decay_rate'])

random_seed = args["random_seed"]

if  args["shuffle"]:
    np.random.seed(random_seed)
    permut = np.random.permutation(len(dataset))
    dataset = dataset[permut]
else:
    pass

division = args["division"]
dataset_train = dataset[:division[0]]
dataset_test = dataset[division[0]:division[0]+division[1]]
dataset_val = dataset[division[0]+division[1]:division[0]+division[1]+division[2]]

epoch = args["max_epoch"]
max_grad = args["max_grad"]
batch_size = args["batch_size"]
grad_clip = args["grad_clip"]

trainer = Train_with_force(model=model,ema_model = ema_model, epoches=epoch, optimizer=optimizer, dataset_train=dataset_train, dataset_test=dataset_test, dataset_val=dataset_val,
    scheduler=scheduler, std=1.0, record_limit=args['record_limit'], grad_clip=grad_clip, max_grad=max_grad, random_seed=random_seed,batch_size=batch_size, device=device, mol_name = mol_name, args = args, force_weight=args["force_weight"])
#






trainer.train()

print(torch.cuda.max_memory_allocated(device = device))