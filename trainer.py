import torch
import os
import json
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
import datetime
import time

class Train_EMA:
    def __init__(self, model, ema_model, args, epoches, dataset, division, optimizer, scheduler,
         std, grad_clip = True, max_grad = 10.0, random_seed = 41, device = 'cuda', batch_size = 128):
        self.model = model.to(device)
        self.ema_model = ema_model
        self.epoches = epoches
        self.optimizer = optimizer
        self.clip = grad_clip
        self.scheduler = scheduler
        self.max_norm = max_grad
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.permutation = np.random.permutation(len(dataset))
        self.dataset = dataset[self.permutation]
        self.test_loader = DataLoader(self.dataset[:division[0]], batch_size = batch_size, shuffle=False)
        self.val_loader = DataLoader(self.dataset[division[0]:division[1]], batch_size = batch_size, shuffle=False)
        self.train_loader = DataLoader(self.dataset[division[1]:], batch_size = batch_size, shuffle=False)
        self.device = device
        self.std = std
        self.batch_size = batch_size
        self.args = args

    def epoch(self):
        self.model.train()
        loss_epoch = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            results = self.model(data)
            loss = F.smooth_l1_loss(results, data.y)   # loss_function
            loss.backward()
            if self.clip:
                clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.max_norm)
            loss_epoch += loss.item() * data.num_graphs
            self.optimizer.step()
            self.scheduler.step()
            self.ema_model.update_parameters(self.model)
        
        return loss_epoch/len(self.train_loader.dataset)
    
    def test(self,loader):
        self.model.eval()
        error = 0
        for data in loader:
            data = data.to(self.device)
            error += self.std*(self.ema_model(data)  - data.y ).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    def train(self):
        best_val_error = None
        test_error = None
        
        # define a log file
        num_of_run = 1
        log_file_name = 'DBGNNLogRUN'
        log_path = os.getcwd()+'/log/'+log_file_name + str(num_of_run) + '.log'
        format_sentence = "{time_str}[lr]:{lr:.7f}\t[epoch]:{epoch:03d}\t[Loss]:{loss:.7f}\t[ValMAE]:{val_error:.7f}\t[TestMAE]:{test_error:.7f}\n"
        while os.path.exists(log_path):
            num_of_run = num_of_run+1
            log_path = os.getcwd()+'/log/'+log_file_name + str(num_of_run) + '.log'
        with open(log_path,'a') as f:
            f.write(f'clip:{self.clip}\tmax_grad:{self.max_norm:.7f}\trandom_seed:{self.random_seed:03d}\tstatistics:{self.dataset.data.y.mean():.7f}\t'
                f'training sample: {len(self.train_loader.dataset):03d}\tval_sample: {len(self.val_loader.dataset):03d}\tbatch_size:{self.batch_size:03d} \n')
        print(log_path)
        print(type(self.model))

        save_dir = "./modelsaves/"+f'/{log_file_name}{str(num_of_run)}/'
        save_file = save_dir + 'ckpt/ckpt_best.pth'

        if not os.path.isdir(save_dir+'/ckpt'):
            os.mkdir(save_dir)
            os.mkdir(save_dir+'/ckpt')

        with open(f'{save_dir}args.json','w')as f:
            json.dump(self.args, f, indent=1)

        # training loop
        for epoch in range(self.epoches):
            lr = self.scheduler.optimizer.param_groups[0]['lr']
            loss = self.epoch()
            val_error = self.test(self.val_loader)

            # save ckpt
            if (best_val_error is None or val_error <= best_val_error):
                best_val_error = val_error
                if epoch > 100:
                    test_error = self.test(self.test_loader)
                    checkpoint = {"model": self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler.state_dict(),
                                    'epoch': epoch}
                    if os.path.isfile(save_file):
                        os.remove(save_file)
                    torch.save(checkpoint, save_file)
                    print('current ckpt epoch:%s' %epoch)
        
            # write log file per epoch
            print(f'Epoch: {epoch+1:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
                    f'Val MAE: {val_error:.7f}, best_Val_MAE:{best_val_error:.7f}, Test MAE: {test_error}')
            if not test_error:
                test_error = -1.0
            log_line = format_sentence.format(time_str = time.strftime("%m_%d_%H_%M_%S"), lr = lr, epoch = epoch+1, loss = loss, val_error = val_error, test_error =test_error)
            with open(log_path, 'a') as f:
                f.write(log_line)
            test_error = None

