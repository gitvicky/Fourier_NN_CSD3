#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:57:06 2021

@author: vgopakum

Fourier RNN on the JOREK MHD 
Mark 1 setup. 
https://www.notion.so/engine-out-completely/Fourier-NNs-9d45170203ae419795584cf24d142423
"""

# %%

import wandb
configuration = {"Type": 'Elman RNN',
                 "Case": 'RBB Camera',
                 "Calibration": 'Calcam',
                 "Epochs": 100,
                 "Batch Size": 2,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.01,
                 "Scheduler Step": 100 ,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'ReLU',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10, 
                 "T_out": 50,
                 "Step": 1,
                 "Modes":32,
                 "Width": 64,
                 "Hidden Size":32,
                 "Cells": 1,
                 "Variables": 1,
                 "Resolution":1, 
                 "Noise":0.0}


run = wandb.init(project='Fourier RNN',
                 notes='LP loss',
                 config=configuration,
                 mode = 'online'
                ) 

run_id = wandb.run.id

wandb.save('FRNN_rbb.py')

# %%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm 

# from pytorch_model_summary import summary

import os 
import sys
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import time
# from utilities3 import *

torch.manual_seed(0) 
np.random.seed(0)

path = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.getcwd()))
# model_loc = os.path.dirname(os.path.dirname(os.getcwd()))
model_loc = os.getcwd()


# frnn_loc = os.path.dirname(os.getcwd())
# sys.path.insert(0, frnn_loc)


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#################################################
#
# Utilities
#
#################################################

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class MinMax_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

#Adding Gaussian Noise to the training dataset
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = torch.FloatTensor([mean])
        self.std = torch.FloatTensor([std])
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

additive_noise = AddGaussianNoise(0.0, configuration['Noise'])
additive_noise.cuda()
# %%

################################################################
# Loading Data 
################################################################
#Importing the Data 

# %%
data =  np.load(data_loc + '/Data/Cam_Data/rbb_fno_data.npy')
data_calib =  np.load(data_loc + '/Data/Cam_Data/rbb_rz_pos.npz')


# %%
#
res = configuration['Resolution']
u_sol = data.astype(np.float32)[:,:,::res, ::res]

grid_size_x = u_sol.shape[2]
grid_size_y = u_sol.shape[3]

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)


ntrain = 50
ntest = 9

S_x = grid_size_x #Grid Size
S_y = grid_size_y #Grid Size

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']

batch_size = configuration['Batch Size']
batch_size2 = batch_size

hidden_size = configuration['Hidden Size']
num_cells = configuration['Cells']

t1 = default_timer()


T_in = configuration['T_in']
T = configuration['T_out']
T_out = T
step = configuration['Step']
################################################################
# load data
################################################################

train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

test_a = u[-ntest:,:,:,:T_in]
test_u = u[-ntest:,:,:,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)


# %%
a_normalizer = UnitGaussianNormalizer(train_a)
# a_normalizer = RangeNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
# y_normalizer = RangeNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)


# %%

train_a = train_a.reshape(ntrain,S_x,S_y,T_in)
test_a = test_a.reshape(ntest,S_x,S_y,T_in)

'''
#Using arbitrary R and Z positions sampled uniformly within a specified domain range. 
# pad the location (x,y)
x = np.linspace(0.5, 1.5, 448)[::res]
gridx = torch.tensor(x, dtype=torch.float)
gridx = gridx.reshape(1, S_x, 1, 1).repeat([1, 1, S_y, 1])

y = np.linspace(-1.0, 1.0, 640)[::res]
gridy = torch.tensor(y, dtype=torch.float)
gridy = gridy.reshape(1, 1, S_y, 1).repeat([1, S_x, 1, 1])

'''
#Using the calibrated R and Z positions averaged over the time and shots. 
gridx = data_calib['r_pos'][::res, ::res]
gridy = data_calib['z_pos'][::res, ::res]
gridx = torch.tensor(gridx, dtype=torch.float)
gridy = torch.tensor(gridy, dtype=torch.float)
gridx = gridx.reshape(1, S_x, S_y, 1)
gridy = gridy.reshape(1, S_x, S_y, 1)



train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1])), dim=-1)
test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1])), dim=-1)

gridx = gridx.to(device)
gridy = gridy.to(device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)


# %%
################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FRNN_Cell_PT(nn.Module):
   def __init__(self, modes, width, batch_first=True):
        super(FRNN_Cell_PT, self).__init__()
        
        self.modes = modes
        self.width = width
        
        self.F_x = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_h = SpectralConv2d(self.width, self.width, self.modes, self.modes)

        self.W_x = nn.Conv1d(self.width, self.width, 1)
        self.W_h = nn.Conv1d(self.width, self.width, 1)
        self.bn_x = torch.nn.BatchNorm2d(self.width)
        self.bn_h = torch.nn.BatchNorm2d(self.width)
        

        
   def forward(self, x, h):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        h = h.permute(0, 3, 2, 1)
        h1 = self.F_h(h)
        h2 = self.W_h(h.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_y, size_x)
        h = self.bn_h(h1+h2)
        # h = h1 + h2
        h = F.relu(h)
        h = h.permute(0, 2, 3, 1)
        
        x = x.permute(0, 3, 2, 1)
        x1 = self.F_x(x)
        x2 = self.W_x(x.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_y, size_x)
        x = self.bn_x(x1+x2)
        # x = x1 + x2
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        
        h = h+x
        y = torch.tanh(h)
        
        return y, h.clone().detach()
    
   def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
    
        return c


class FRNN(nn.Module):
   def __init__(self, modes, width, n_output, n_hidden, n_cells, T_in, batch_first=True):
        super(FRNN, self).__init__()
        
        self.modes = modes
        self.width = width
        self.n_output = n_output 
        self.n_hidden = n_hidden
        
        self.linear_in_x = nn.Linear(T_in+2, self.width)
        self.linear_in_h = nn.Linear(self.n_hidden, self.width)
        self.linear_out_x = nn.Linear(self.width, self.n_output)
        self.linear_out_h = nn.Linear(self.width , self.n_hidden)

        self.FRNN_Cells = nn.ModuleList()
        
        for ii in range(n_cells):
            self.FRNN_Cells.append(FRNN_Cell_PT(self.modes, self.width))

        
   def forward(self, x, h):
       
        h = self.linear_in_h(h)
        x = self.linear_in_x(x)

        for cell in self.FRNN_Cells:
            x, h = cell(x, h)

        y = self.linear_out_x(x)
        h = self.linear_out_h(h)
        return y, h.clone().detach()
    
   def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
    
        return c
        
        
model = FRNN(modes, width, output_size, hidden_size, num_cells, T_in).to(device)
wandb.run.summary['Number of Params'] = model.count_params()
# model = nn.DataParallel(model, device_ids = [0,1])
model.to(device)

# wandb.watch(model, log='all')

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
myloss = LpLoss(size_average=False)
# criterion = torch.nn.MSELoss()
criterion = myloss


epochs = configuration['Epochs']
y_normalizer.cuda()

start_time = time.time()
for it in tqdm(range(epochs)):
    train_loss = 0
    for inp, out in train_loader:
        inp, out = inp.to(device), out.to(device)
        inp = additive_noise(inp)

        loss = 0 
        # hidden = torch.ones((inp.shape[0], grid_size, grid_size, hidden_size)).to(device)*inp[:,:,:,0:1]
        # hidden = torch.zeros((inp.shape[0], grid_size, grid_size, hidden_size )).to(device)
        
        hidden =torch.cat((torch.ones((inp.shape[0], grid_size_x, grid_size_y, hidden_size-2)).to(device)*inp[:,:,:,0:1], 
                    gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1).to(device)
        
        for t in range(0, T_out, step):
            
            o = out[..., t:t + step]
            model_out, hidden = model(inp, hidden)
            hidden = torch.cat((hidden[...,:-2], 
                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1).to(device)
            
            loss += criterion(model_out, o)
            
            if t == 0:
                pred = model_out
            else:
                pred = torch.cat((pred, model_out), -1)

            inp = torch.cat((inp[..., step:-2],model_out,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss
    scheduler.step()
    
    print("Epoch: %.d, Train Error: %.3e" % (it, train_loss))
    wandb.log({'Loss': loss})


train_time = time.time() - start_time  
# %%
batch_size = 1 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape).to(device)
index = 0

with torch.no_grad():
    for inp, out in test_loader:
        inp, out = inp.to(device), out.to(device)
        inp = additive_noise(inp)
        # hidden = torch.ones((inp.shape[0], grid_size, grid_size, hidden_size)).to(device)*inp[:,:,:,0:1]
        # hidden = torch.zeros((1, grid_size, grid_size, hidden_size)).to(device)
                
        hidden =torch.cat((torch.ones((inp.shape[0], grid_size_x, grid_size_y, hidden_size-2)).to(device)*inp[:,:,:,0:1], 
                        gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1).to(device)
        
        for t in range(0, T_out, step):
            o = out[..., t:t + step]
            model_out, hidden = model(inp, hidden)

            hidden = torch.cat((hidden[...,:-2], 
                        gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1).to(device)
            
            if t == 0:
                pred = model_out
            else:
                pred = torch.cat((pred, model_out), -1)   
            

            inp = torch.cat((inp[..., step:-2], model_out,
                            gridx.repeat([1, 1, 1, 1]), gridy.repeat([1, 1, 1, 1])), dim=-1)   

        pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
    
test_l2 = (pred_set - test_u.to(device)).pow(2).mean()
print('Fourier RNN - Testing Error: %.3e' % (test_l2))

wandb.run.summary['Training Time'] = train_time
wandb.run.summary['Test Error'] = test_l2

torch.save(model.state_dict(), model_loc + '/Models/FRNN_rbb_'+run_id + '.pth')
wandb.save(model_loc + '/Models/FRNN_rbb_'+run_id + '.pth')

# %%


# %%
idx = np.random.randint(0, ntest) 
u_field = test_u[idx].cpu().detach().numpy()

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.title.set_text('Initial')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2,3,2)
ax.imshow(u_field[:,:,int(T_out/2)], cmap=cm.coolwarm)
ax.title.set_text('Middle')

ax = fig.add_subplot(2,3,3)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)
ax.title.set_text('Final')


u_field = pred_set[idx].cpu().detach().numpy()

ax = fig.add_subplot(2,3,4)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.set_ylabel('F-RNN')

ax = fig.add_subplot(2,3,5)
ax.imshow(u_field[:,:,int(T_out/2)], cmap=cm.coolwarm)

ax = fig.add_subplot(2,3,6)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)

wandb.log({"RBB Camera" : plt})
wandb.run.finish()

