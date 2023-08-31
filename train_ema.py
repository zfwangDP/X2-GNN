import torch
import torch.nn as nn
import numpy as np
from biqm9u0_6 import BIQM9U0_6
from torch_geometric.loader import DataLoader
from trainer import Train_EMA
from xgnn import xgnn_poly
from scheduler import LinearWarmupExponentialDecay

torch.set_num_threads(1)

dataset = BIQM9U0_6(input_sdf='./sdfs/qm9U0.sdf')

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = dataset.data.y*0.04336414   #converte kcal/mol to eV
mean, std = mean.item(), std.item()
print("statistics:",mean,std)

device = 'cuda'
model = xgnn_poly(conv_layers = 4, sbf_dim=7, rbf_dim=6, in_channels = 128, K = 2, heads = 16, embedding_size = 128).to(device)

ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
        0.99 * averaged_model_parameter + 0.01 * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad = False)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, min_lr=0.000001)
scheduler = LinearWarmupExponentialDecay(optimizer, warmup_steps = 3000,decay_steps=3000000, decay_rate = 0.01)

batch_size = 16
conv_layers = 4
dim = 256
K=2
sbf_dim = 16
heads = 2
random_seed = 42
epoch = 1
grad_clip = True
max_grad = 1000
division = [1000,2000]

trainer = Train_EMA(model=model,ema_model = ema_model, epoches=epoch, dataset=dataset, division=division, optimizer=optimizer, 
    scheduler=scheduler, std=(1/0.04336414), grad_clip=grad_clip, max_grad=max_grad, random_seed=random_seed, batch_size=batch_size)

trainer.train()

print(torch.cuda.max_memory_allocated(device = 'cuda'))