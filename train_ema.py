import torch
import json
from qm9_allprop import QM9_allprop
from trainer import Train_EMA
from xgnn import xgnn_poly, xgnn_poly_global
from scheduler import LinearWarmupExponentialDecay
from torch_scatter import scatter_add

labels = {0:'dipole',1:'polarizability',2:'HOMO',3:'LUMO',4:'GAP',5:'spatial extent',6:'zpve',7:'U0',8:'U',9:'H',10:'G',11:'Cv'}
atom_ref = torch.zeros(12,10)
atom_ref[7] = torch.tensor([torch.nan,-0.500273,torch.nan,torch.nan,torch.nan,torch.nan,
                            -37.846772,-54.583861,-75.064579,-99.718730])
atom_ref[8] = torch.tensor([torch.nan,-0.498857,torch.nan,torch.nan,torch.nan,torch.nan,
                            -37.845355,-54.582445,-75.063163,-99.717314])
atom_ref[9] = torch.tensor([torch.nan,-0.497912,torch.nan,torch.nan,torch.nan,torch.nan,
                            -37.844411,-54.581501,-75.062219,-99.716370])
atom_ref[10] = torch.tensor([torch.nan,-0.510927,torch.nan,torch.nan,torch.nan,torch.nan,
                            -37.861317,-54.598897,-75.079532,-99.733544])
atom_ref[11] = torch.tensor([torch.nan,2.981,torch.nan,torch.nan,torch.nan,torch.nan,
                            2.981,2.981,2.981,2.981])

args = {}
with open('./config.json','rt') as f:
    args.update(json.load(f))

torch.set_num_threads(args['num_thread'])

dataset = QM9_allprop(input_file=args['input_file'])
target = args['target']
atom_affi = torch.arange(len(dataset)).repeat_interleave(dataset.slices['x'][1:] - dataset.slices['x'][:-1])
mol_ref = scatter_add(atom_ref[target][dataset.data.x],index=atom_affi,dim=0)
dataset.data.y = dataset.data.y[:,target].squeeze() - mol_ref

if target in [2,3,4,6,7,8,9,10]:
    dataset.data.y = dataset.data.y * 27.211385056
    print_calibration = 1/0.04336414
else:
    print_calibration = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if target in [6,7,8,9,10,11]:
    model = xgnn_poly(conv_layers=args['conv_layers'], sbf_dim=args['sbf_dim'], rbf_dim=args['rbf_dim'], in_channels=args['in_channels'], heads=args['heads'], embedding_size=args['embedding_size'], device=device).to(device)
else:
    model = xgnn_poly_global(conv_layers=args['conv_layers'], sbf_dim=args['sbf_dim'], rbf_dim=args['rbf_dim'], in_channels=args['in_channels'], heads=args['heads'], embedding_size=args['embedding_size'], device=device, pool_option=args['pool_option']).to(device)
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
        args['ema_decay'] * averaged_model_parameter + (1-args['ema_decay']) * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
optimizer = torch.optim.Adam(model.parameters(), lr=args['max_lr'], amsgrad = False)
assert args['scheduler'] in ["LinearWarmupExponentialDecay","ReduceLROnPlateau"], 'unimplemented LR scheduler'
if args['scheduler']=="LinearWarmupExponentialDecay":
    scheduler = LinearWarmupExponentialDecay(optimizer, warmup_steps = args['warmup_steps'],decay_steps=args['decay_steps'], decay_rate = args['decay_rate'])
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args['reduce_factor'], patience=args['patience'], min_lr=args['max_lr']*args['decay_rate'])


trainer = Train_EMA(model=model, ema_model = ema_model, args=args, epoches=args['max_epoch'], dataset=dataset, division=args['division'], optimizer=optimizer, scheduler=scheduler, std=print_calibration, grad_clip=args['grad_clip'], max_grad=args['max_grad'],random_seed=args['random_seed'], batch_size=args['batch_size'])
trainer.train()

print(torch.cuda.max_memory_allocated(device = 'cuda'))
