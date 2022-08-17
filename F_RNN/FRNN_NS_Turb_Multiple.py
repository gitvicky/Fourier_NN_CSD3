#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:34:57 2022


@author: vgopakum

FRNN with NS Cylinder - Vortex Shedding - Data contains different Reynolds number ranging from ~
Obtained by changing the Kinematic Viscosity. 

"""
# %%
import wandb
configuration = {"Type": 'Elman RNN',
                 "Case": 'NS Cyl-UVP',
                #  "Field": 'P',
                 "Epochs": 500,
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.01,
                 "Scheduler Step": 100 ,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'ReLU',
                 "Normalisation Strategy": 'Gaussian',
                 "T_in": 20, 
                 "T_out": 20,
                 "Step": 1,
                 "Modes":16,
                 "Width": 32,
                 "Hidden Size":32,
                 "Cells": 2,
                 "Variables": 3,
                 "Noise":0.5}


run = wandb.init(project='Fourier RNN',
                 notes='',
                 config=configuration,
                 mode='online')

run_id = wandb.run.id

wandb.save('FRNN_NS_Turb_Multiple.py')


# %%

import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 

import operator
from functools import reduce
from functools import partial

import time 
from timeit import default_timer
from tqdm import tqdm 

torch.manual_seed(0)
np.random.seed(0)

import os 
path = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.getcwd()))
model_loc = os.getcwd()

# %%


#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


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
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

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

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

# %%


#Complex multiplication
def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


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
# %%
################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, num_vars, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, num_vars, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivxy,iovxy->bovxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, num_vars,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FRNN_Cell_PT(nn.Module):
   def __init__(self, modes, width, batch_first=True):
        super(FRNN_Cell_PT, self).__init__()
        
        self.modes = modes
        self.width = width
        
        self.F_x = SpectralConv2d_fast(self.width, self.width, self.modes, self.modes)
        self.F_h = SpectralConv2d_fast(self.width, self.width, self.modes, self.modes)

        self.W_x = nn.Conv1d(self.width, self.width, 1)
        self.W_h = nn.Conv1d(self.width, self.width, 1)
        self.bn_x = torch.nn.BatchNorm3d(self.width)
        self.bn_h = torch.nn.BatchNorm3d(self.width)
        

        
   def forward(self, x, h):
       
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # h = h.permute(0, 3, 2, 1)
        h = h.permute(0, 4, 1, 2, 3)
        h1 = self.F_h(h)
        h2 = self.W_h(h.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, num_vars, size_x, size_y)
        h = self.bn_h(h1+h2)
        # h = h1 + h2
        h = F.relu(h)
        # h = h.permute(0, 2, 3, 1)
        h = h.permute(0, 2, 3, 4, 1)

        
        # x = x.permute(0, 3, 2, 1)
        x = x.permute(0, 4, 1, 2, 3)
        x1 = self.F_x(x)
        x2 = self.W_x(x.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, num_vars, size_x, size_y)
        x = self.bn_x(x1+x2)
        # x = x1 + x2
        x = F.relu(x) 
        # x = x.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 4, 1)
        
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
        
        

################################################################
# Loading Data 
################################################################

# %%
data =  np.load(data_loc + '/Data/NS_Cyl_FNO_turb.npz')


# %%
#
u_sol = data['U'].astype(np.float32)
v_sol = data['V'].astype(np.float32)
p_sol = data['P'].astype(np.float32)

x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
grid_size = len(x)

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

v = torch.from_numpy(v_sol)
v = v.permute(0, 2, 3, 1)

p = torch.from_numpy(p_sol)
p = p.permute(0, 2, 3, 1)

uvp = torch.stack((u,v,p), dim=1)

ntrain = 80
ntest = 20
S = 99 #Grid Size

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']
num_vars = configuration['Variables']

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

train_a = uvp[:ntrain,:,:,:,:T_in]
train_u = uvp[:ntrain,:,:,:,T_in:T+T_in]

test_a = uvp[-ntest:,:,:,:,:T_in]
test_u = uvp[-ntest:,:,:,:,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)


# %%
a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)
# %%

train_a = train_a.reshape(ntrain,num_vars,S,S,T_in)
test_a = test_a.reshape(ntest,num_vars, S,S,T_in)

# pad the location (x,y)
gridx = torch.tensor(x, dtype=torch.float)
gridx = gridx.reshape(1, 1, S, 1, 1).repeat([1, num_vars, 1, S, 1])
gridy = torch.tensor(y, dtype=torch.float)
gridy = gridy.reshape(1, 1, 1, S, 1).repeat([1, num_vars, S, 1, 1])

train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1])), dim=-1)
test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1])), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# training and evaluation
################################################################

model = FRNN(modes, width, output_size, hidden_size, num_cells, T_in).to(device)
model.to(device)

# wandb.watch(model, log='all')
wandb.run.summary['Number of Params'] = model.count_params()


print("Number of model params : " + str(model.count_params()))

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
myloss = LpLoss(size_average=False)
criterion = torch.nn.MSELoss()

myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

# %%

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
        
        hidden =torch.cat((torch.ones((inp.shape[0], num_vars, grid_size, grid_size, hidden_size-2)).to(device)*inp[:,:,:,:,0:1], 
                    gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1])), dim=-1).to(device)
        
        for t in range(0, T_out, step):
            
            o = out[..., t:t + step]
            model_out, hidden = model(inp, hidden)
            hidden = torch.cat((hidden[...,:-2], 
                gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1])), dim=-1).to(device)
            
            loss += criterion(model_out, o)
            
            if t == 0:
                pred = model_out
            else:
                pred = torch.cat((pred, model_out), -1)

            inp = torch.cat((inp[..., step:-2],model_out,
                            gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1])), dim=-1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss
    scheduler.step()
    
    print("Epoch: %.d, Train Error: %.3e" % (it, train_loss))
    wandb.log({'Loss': loss})


train_time = time.time() - start_time  
# %%

torch.save(model.state_dict(), model_loc + '/Models/FRNN_NS_cyl_UVP_'+run_id + '.pth')
wandb.save(model_loc + '/Models/FRNN_NS_cyl_UVP_'+run_id + '.pth')


# model = torch.load('NS_FNO_2d_time.pth', map_location=torch.device('cpu'))

# %%

#Testing 
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
                
        hidden =torch.cat((torch.ones((inp.shape[0], num_vars, grid_size, grid_size, hidden_size-2)).to(device)*inp[:,:,:,:,0:1], 
                        gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1])), dim=-1).to(device)
        
        for t in range(0, T_out, step):
            o = out[..., t:t + step]
            model_out, hidden = model(inp, hidden)

            hidden = torch.cat((hidden[...,:-2], 
                        gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1])), dim=-1).to(device)
            
            if t == 0:
                pred = model_out
            else:
                pred = torch.cat((pred, model_out), -1)   
            

            inp = torch.cat((inp[..., step:-2], model_out,
                            gridx.repeat([1, 1, 1, 1, 1]), gridy.repeat([1, 1, 1, 1, 1])), dim=-1)   

        pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
    
test_l2 = (pred_set - test_u.to(device)).pow(2).mean()

print('Testing Error: %.3e' % (test_l2))
    
wandb.run.summary['Training Time'] = train_time
wandb.run.summary['Test Error'] = test_l2


# %%
idx = np.random.randint(0,ntest) 
u_field = test_u[idx][0]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.title.set_text('Initial')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2,3,2)
ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm)
ax.title.set_text('Middle')

ax = fig.add_subplot(2,3,3)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)
ax.title.set_text('Final')

u_field = pred_set[idx][0].cpu()

ax = fig.add_subplot(2,3,4)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.set_ylabel('FRNN')

ax = fig.add_subplot(2,3,5)
ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm)

ax = fig.add_subplot(2,3,6)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)


wandb.log({"NS Vortex - U": plt})

# idx = np.random.randint(0,ntest) 
u_field = test_u[idx][1]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.title.set_text('Initial')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2,3,2)
ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm)
ax.title.set_text('Middle')

ax = fig.add_subplot(2,3,3)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)
ax.title.set_text('Final')

u_field = pred_set[idx][1].cpu()

ax = fig.add_subplot(2,3,4)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.set_ylabel('FRNN')

ax = fig.add_subplot(2,3,5)
ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm)

ax = fig.add_subplot(2,3,6)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)


wandb.log({"NS Vortex - V": plt})

u_field = test_u[idx][2]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.title.set_text('Initial')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2,3,2)
ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm)
ax.title.set_text('Middle')

ax = fig.add_subplot(2,3,3)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)
ax.title.set_text('Final')

u_field = pred_set[idx][2].cpu()

ax = fig.add_subplot(2,3,4)
ax.imshow(u_field[:,:,0], cmap=cm.coolwarm)
ax.set_ylabel('FRNN')

ax = fig.add_subplot(2,3,5)
ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm)

ax = fig.add_subplot(2,3,6)
ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)


wandb.log({"NS Vortex - P": plt})

wandb.run.finish()
