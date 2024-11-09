import torch
import os
import json
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from scheduler import LinearWarmupExponentialDecay
import time
from torch.utils.data import ConcatDataset

def resolve_loss_funtion(loss_function_name):
    assert loss_function_name in ["smooth_l1","l1","l2","mae","mse"], "unsupported loss type"
    if loss_function_name=="smooth_l1":
        return F.smooth_l1_loss
    elif loss_function_name in ["l1",'mae']:
        return F.l1_loss
    else:
        return F.mse_loss

class Train_EMA:
    def __init__(self, model, ema_model, args, epoches, dataset, division, optimizer, scheduler, std, shuffle=True,loss_cuntion="smooth_l1", grad_clip = True, max_grad = 10.0, random_seed = 41, device = 'cuda', batch_size = 128, record_limit=50, datasets = None):
        self.model = model.to(device)   # register model
        self.ema_model = ema_model  # register ema_model
        self.epoches = epoches  # max epoches to train
        self.optimizer = optimizer  # register optimizer
        self.clip = grad_clip   # if to clip the gradient(by its norm(l2))
        self.scheduler = scheduler  # register scheduler
        self.max_norm = max_grad    # the max norm of gradient after clipping(only effective when grad_clip is True)
        self.random_seed = random_seed  # set the random seed to shuffle the dataset
        self.shuffle=shuffle    # if to shuffle the dataset before when initializing
        self.datasets = datasets
        if not datasets:
            if shuffle:
                np.random.seed(random_seed)
                self.permutation = np.random.permutation(len(dataset))
                self.dataset = dataset[self.permutation]
                self.test_loader = DataLoader(self.dataset[:division[0]], batch_size = batch_size, shuffle=False)
                self.val_loader = DataLoader(self.dataset[division[0]:division[1]], batch_size = batch_size, shuffle=False)
                self.train_loader = DataLoader(self.dataset[division[1]:], batch_size = batch_size, shuffle=False)
            else:
                #self.dataset,self.dataset_val,self.dataset_test = dataset
                self.dataset = dataset
                self.test_loader = DataLoader(self.dataset[:division[0]], batch_size = batch_size, shuffle=False)
                self.val_loader = DataLoader(self.dataset[division[0]:division[1]], batch_size = batch_size, shuffle=False)
                self.train_loader = DataLoader(self.dataset[division[1]:], batch_size = batch_size, shuffle=False)
        else:
            self.dataset_train, self.dataset_val, self.dataset_test = datasets
            self.test_loader = DataLoader(self.dataset_test, batch_size = batch_size, shuffle=False)
            self.val_loader = DataLoader(self.dataset_val, batch_size = batch_size, shuffle=False)
            self.train_loader = DataLoader(self.dataset_train, batch_size = batch_size, shuffle=True)

        self.device = device    # declare the device(used when mv data)
        self.std = std  # when first designed it is used to normalize the labels, now only used to calibrate the MAE unit to kcal/mol
        self.loss_function = resolve_loss_funtion(loss_cuntion)
        self.batch_size = batch_size    # declare batch_size, used for logging
        self.record_limit = record_limit    # epoch start to save ckpt file
        self.args = args    # other args passed in
        if isinstance(scheduler, LinearWarmupExponentialDecay): # maybe more choice in the future
            self.scheduler_per_step = True
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler_per_step = False
        else:
            raise TypeError("Unsupported Scheduler Type")
        if self.args['save_all_ckpts']:
            self.record_limit = 0

    def epoch(self):
        self.model.train()
        loss_epoch = 0.

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            results = self.model(data)
            loss = self.loss_function(results, data.y)   # loss_function
            loss.backward()
            if self.clip:
                clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.max_norm)
            loss_epoch += loss.item() * data.num_graphs
            self.optimizer.step()
            self.scheduler.step()
            self.ema_model.update_parameters(self.model)
        
        return loss_epoch/len(self.train_loader.dataset)

    def epoch_OP(self):
        self.model.train()
        loss_epoch = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            results = self.model(data)
            loss = self.loss_function(results, data.y)
            loss.backward()
            if self.clip:
                clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.max_norm)
            loss_epoch += loss.item() * data.num_graphs
            self.optimizer.step()
            self.ema_model.update_parameters(self.model)
        
        return loss_epoch/len(self.train_loader.dataset)

    def epoch_bde_all(self):
        self.model.train()
        loss_epoch = 0.
        all_bdr_num = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            results = self.model(data)
            num_bdr = (len(set(data.bde_idx[data.is_cleave].tolist()))*2-2)
            all_bdr_num += num_bdr
            loss = ((results - F.relu(data.cleave_en[data.is_cleave]))/data.bde_num[data.is_cleave]).abs().sum()/num_bdr   # weighted mae loss, averaged by each bd reaction
            loss.backward()
            if self.clip:
                clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.max_norm)
            loss_epoch += loss.item() * num_bdr
            self.optimizer.step()
            self.scheduler.step()
            self.ema_model.update_parameters(self.model)
        
        return loss_epoch/all_bdr_num

    def test(self,loader):
        self.model.eval()
        error = 0
        for data in loader:
            data = data.to(self.device)
            error += self.std*(self.ema_model(data)  - data.y ).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    def test_bde_all(self, loader):
        self.model.eval()
        error = 0
        num_bdr = 0
        for data in loader:
            data = data.to(self.device)
            error += self.std*((self.ema_model(data)  - data.cleave_en[data.is_cleave])/data.bde_num[data.is_cleave]).abs().sum().item()
            #num_bdr += (len(set(data.bde_idx[data.is_cleave]))*2-2)
            num_bdr += (len(set(data.bde_idx[data.is_cleave].tolist()))*2-2)
        return error / num_bdr

    def epoch_vectorial_OP(self):
        self.model.train()
        loss_epoch = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            predicted_position = self.model(data)
            bond_lengths = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]).float(), dim = 1)
            predicted_bond_lengths = torch.norm(predicted_position[data.edge_index[0]] - predicted_position[data.edge_index[1]], dim = 1)
            loss = self.loss_function(predicted_bond_lengths, bond_lengths)
            loss.backward()
            if self.clip:
                clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.max_norm)
            loss_epoch += loss.item() * data.num_graphs
            self.optimizer.step()
            self.ema_model.update_parameters(self.model)
        
        return loss_epoch/len(self.train_loader.dataset)

    def test_vectorial(self, loader):
        self.model.eval()
        error = 0
        num_bonds = 0
        for data in loader:
            data = data.to(self.device)
            predicted_position = self.model(data)
            bond_lengths = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]).float(), dim = 1)
            predicted_bond_lengths = torch.norm(predicted_position[data.edge_index[0]] - predicted_position[data.edge_index[1]], dim = 1)
            error += ((bond_lengths  - predicted_bond_lengths)).abs().sum().item()
            num_bonds += bond_lengths.shape[0]
        return error / num_bonds

    def train(self):
        best_val_error = None
        test_error = None
        
        # define a log file
        num_of_run = 1
        log_file_name = 'DBGNNLogRUN'
        log_path = os.getcwd()+'/log/qm9/'+log_file_name + str(num_of_run) + '.log'
        format_sentence = "{time_str}[lr]:{lr:.7f}\t[epoch]:{epoch:03d}\t[Loss]:{loss:.7f}\t[ValMAE]:{val_error:.7f}\t[TestMAE]:{test_error:.7f}\n"
        while os.path.exists(log_path):
            num_of_run = num_of_run+1
            log_path = os.getcwd()+'/log/qm9/'+log_file_name + str(num_of_run) + '.log'
        with open(log_path,'a') as f:
            if  self.datasets is not None:
                f.write(f'clip:{self.clip}\tmax_grad:{self.max_norm:.7f}\trandom_seed:{self.random_seed:03d}\tstatistics:Disabled\tstatistics:ConcatDataset\t'
                f'training sample: {len(self.train_loader.dataset):03d}\tval_sample: {len(self.val_loader.dataset):03d}\ttest_sample: {len(self.test_loader.dataset):03d}\tbatch_size:{self.batch_size:03d} \n')
            else:
                if self.shuffle:
                    f.write(f'clip:{self.clip}\tmax_grad:{self.max_norm:.7f}\trandom_seed:{self.random_seed:03d}\tstatistics:{self.dataset.data.y.mean():.7f}\t'
                    f'training sample: {len(self.train_loader.dataset):03d}\tval_sample: {len(self.val_loader.dataset):03d}\ttest_sample: {len(self.test_loader.dataset):03d}\tbatch_size:{self.batch_size:03d} \n')
                else:
                    f.write(f'clip:{self.clip}\tmax_grad:{self.max_norm:.7f}\trandom_seed:{self.random_seed:03d}\tstatistics:{len(self.dataset):09d}\t'
                    f'training sample: {len(self.train_loader.dataset):03d}\tval_sample: {len(self.val_loader.dataset):03d}\ttest_sample: {len(self.test_loader.dataset):03d}\tbatch_size:{self.batch_size:03d} \n')
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
            if self.args['vectorial']:
                loss = self.epoch_vectorial_OP()
                val_error = self.test_vectorial(self.val_loader)
            elif self.args['bde_all']:
                loss = self.epoch_bde_all()
                val_error = self.test_bde_all(self.val_loader)
            elif self.scheduler_per_step:
                loss = self.epoch()
                val_error = self.test(self.val_loader)
            else:
                loss = self.epoch_OP()
                val_error = self.test(self.val_loader)
            if not self.scheduler_per_step:
                self.scheduler.step(val_error)

            if self.args["save_all_ckpts"]:
                save_file = save_dir + f'ckpt/ckpt_epoch{epoch}.pth'

            # save ckpt
            if (best_val_error is None or val_error <= best_val_error) or self.args["save_all_ckpts"]:
                if (best_val_error is None or val_error <= best_val_error):
                    best_val_error = val_error
                if epoch > self.record_limit:
                    if self.args['vectorial']:
                        test_error = self.test_vectorial(self.test_loader)
                    elif self.args['bde_all']:
                        test_error = self.test_bde_all(self.test_loader)
                    else:
                        test_error = self.test(self.test_loader)
                    checkpoint = {'ema_model':self.ema_model.state_dict(),
                                    "model": self.model.state_dict(),
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


class Train_with_force:
    def __init__(self, model, ema_model, epoches, dataset_train, dataset_test, dataset_val, optimizer, scheduler, mol_name, args, std, record_limit = 50, force_weight = 0.999, grad_clip = True, max_grad = 10.0, random_seed = 41, device = 'cuda', batch_size = 128,):
        self.model = model.to(device)
        self.ema_model = ema_model
        self.epoches = epoches
        self.optimizer = optimizer
        self.clip = grad_clip
        self.scheduler = scheduler
        self.max_norm = max_grad
        self.random_seed = random_seed
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.test_loader = DataLoader(self.dataset_test, batch_size = batch_size, shuffle=False)
        self.val_loader = DataLoader(self.dataset_val, batch_size = batch_size, shuffle=False)
        self.train_loader = DataLoader(self.dataset_train, batch_size = batch_size, shuffle=False)
        self.device = device
        self.std = std
        self.record_limit = record_limit
        self.rho = force_weight
        self.batch_size = batch_size
        self.save_file = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
        self.mol_name = mol_name
        self.args = args
        if isinstance(scheduler, LinearWarmupExponentialDecay):
            self.scheduler_per_step = True
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler_per_step = False
        else:
            raise TypeError("Unsupported Scheduler Type")
        self.rmd17_atom_energy = torch.tensor([torch.nan, -13.56842, torch.nan, torch.nan, torch.nan, torch.nan, -1027.26377, -1482.11161, -2038.53292, -2708.61649 ], device = self.device)

    def epoch_OP(self):
        self.model.train()
        loss_epoch = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            results, forces = self.model(data)
            # Schnet type loss
            #loss = 0.01 * F.smooth_l1_loss(results, data.y) + torch.norm((forces - data.force_label),dim=1).mean()  # loss_function
            # dimenet type loss
            #atoms_energy = scatter_add(self.rmd17_atom_energy[data.x.long()], data.batch)
            loss = (1-self.rho) * F.smooth_l1_loss(results, data.y) + self.rho * F.smooth_l1_loss(forces.view(-1), data.force_label.view(-1))
            loss.backward()
            if self.clip:
                clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.max_norm)
            loss_epoch += loss.item() * data.num_graphs
            self.optimizer.step()
            self.ema_model.update_parameters(self.model)
        
        return loss_epoch/len(self.train_loader.dataset)        

    def epoch_LE(self):
        self.model.train()
        loss_epoch = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            results, forces = self.model(data)
            # Schnet type loss
            #loss = 0.01 * F.smooth_l1_loss(results, data.y) + torch.norm((forces - data.force_label),dim=1).mean()  # loss_function
            # dimenet type loss
            #atoms_energy = scatter_add(self.rmd17_atom_energy[data.x.long()], data.batch)
            loss = (1-self.rho) * F.l1_loss(results, data.y) + self.rho * F.l1_loss(forces.view(-1), data.force_label.view(-1))
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
        force_error = 0
        val_loss = 0
        num_nodes = 0
        for data in loader:
            data = data.to(self.device)
            results, forces = self.ema_model(data)
            with torch.no_grad():
                #atoms_energy = scatter_add(self.rmd17_atom_energy[data.x.long()], data.batch)
                error += (results  - data.y ).abs().sum()  # MAE
                force_error += F.l1_loss(forces, data.force_label, reduction = 'mean') * data.num_nodes # everage force error of atom(vector norm) torch.norm((forces - data.force_label),dim=1).mean()*data.num_graphs
                #loss = 0.01 * F.smooth_l1_loss(results, data.y) + torch.norm((forces - data.force_label),dim=1).mean()
                val_loss += (0.01 * F.l1_loss(results, data.y) + 0.99 * F.l1_loss(forces.view(-1), data.force_label.view(-1))) * data.num_graphs
                num_nodes += data.num_nodes
        return self.std*error/len(loader.dataset), self.std*force_error/num_nodes, val_loss/len(loader.dataset)

    def train(self):
        best_val_loss = None
        test_error = None

        # define a log file
        num_of_run = 1
        log_file_name = 'XGNNForceLog'
        log_path = os.getcwd()+'/log/rmd17/'+log_file_name + str(num_of_run) + '.log'
        format_sentence = "{time_str}[lr]:{lr:.7f}\t[epoch]:{epoch:03d}\t[Loss]:{loss:.7f}\t[ValLoss]:{val_loss:.7f}\t[ValMAE]:{val_error:.7f}\t[ForceError]:{force_error:.7f}\t[TestMAE]:{test_error:.7f}\t[TFE]:{test_force_error:.7f}\n"
        while os.path.exists(log_path):
            num_of_run = num_of_run+1
            log_path = os.getcwd()+'/log/rmd17/'+log_file_name + str(num_of_run) + '.log'
        with open(log_path,'a') as f:
            if  isinstance(self.dataset_train, ConcatDataset):
                f.write(f'clip:{self.clip}\tmax_grad:{self.max_norm:.7f}\trandom_seed:{self.random_seed:03d}\tmolecule:{self.mol_name}\tstatistics:ConcatDataset\t'
                f'training sample: {len(self.train_loader.dataset):03d}\tval_sample: {len(self.val_loader.dataset):03d}\ttest_sample: {len(self.test_loader.dataset):03d}\tbatch_size:{self.batch_size:03d} \n')
            else:
                f.write(f'clip:{self.clip}\tmax_grad:{self.max_norm:.7f}\trandom_seed:{self.random_seed:03d}\tmolecule:{self.mol_name}\tstatistics:{self.dataset_train.data.y.mean():.7f}\t'
                f'training sample: {len(self.train_loader.dataset):03d}\tval_sample: {len(self.val_loader.dataset):03d}\ttest_sample: {len(self.test_loader.dataset):03d}\tbatch_size:{self.batch_size:03d} \n')
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
            if self.scheduler_per_step:
                loss = self.epoch_LE()
            else:
                loss = self.epoch_OP()
            val_error, force_error, val_loss = self.test(self.val_loader)
            #test_error = self.test(self.test_loader)
            if not self.scheduler_per_step:
                self.scheduler.step(val_loss)

            # save ckpt
            #save_file = "./modelsaves/"+self.save_file+'/ckpt/ckpt_best.pth'
            if (best_val_loss is None or val_loss <= best_val_loss):
                best_val_loss = val_loss
                if epoch > self.record_limit:
                    test_error, test_force_error, test_loss = self.test(self.test_loader)
                    checkpoint = {"ema_model": self.ema_model.state_dict(),
                                'model':self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'epoch': epoch}
                    if not os.path.isdir("./modelsaves/"+self.save_file+'/ckpt'):
                        os.mkdir("./modelsaves/"+self.save_file)
                        os.mkdir("./modelsaves/"+self.save_file+'/ckpt')
                    if os.path.isfile(save_file):
                        os.remove(save_file)
                    torch.save(checkpoint, save_file)
                    print("saved as:%s" %save_file)
                    print('current ckpt epoch:%s' %epoch)
        
            # write log file per epoch
            print(f'Epoch: {epoch+1:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
                    f'Val MAE: {val_error:.7f}, Val_force_error: {force_error:.7f}, best_Val_MAE:{best_val_loss:.7f}, Test MAE: {test_error}')
            if not test_error:
                test_error = -1.0
                test_force_error = -1.0

            log_line = format_sentence.format(time_str = time.strftime("%m_%d_%H_%M_%S"), lr = lr, epoch = epoch+1,
                                                 loss = loss, val_loss = val_loss, val_error = val_error, force_error = force_error,
                                                   test_error =test_error, test_force_error = test_force_error)
            with open(log_path, 'a') as f:
                f.write(log_line)
            test_error = None
