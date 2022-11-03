#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:34:57 2022


@author: vgopakum

FNO 2d time on RBB Camera Data 


"""

# %%
import wandb
configuration = {
                "Case": 'RBB Camera',
                #  "Case": 'RBB Camera - Moved',
                "Type": 'FNO-3d',
                 "Calibration": 'Arbitrary',
                 "Epochs": 100,
                 "Batch Size": 2,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 20,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'ReLU',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10, 
                 "T_out": 50,
                #  "Step": 1,
                 "Modes":16,
                 "Width": 16,
                 "Variables": 1,
                 "Resolution":1, 
                 "Noise":0.0}

run = wandb.init(project='FNO-Camera',
                 notes='',
                 config=configuration,
                 mode='online')

run_id = wandb.run.id

wandb.save('FNO_rbb_3d.py')

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

torch.manual_seed(0)
np.random.seed(0)

import os 
path = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.getcwd()))
model_loc = os.getcwd()



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

# %%


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3 

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(T_in+3, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)
        
        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        # x = F.gelu(x)

        x = x.permute(0, 2, 3, 4, 1) 

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x



class Net3d(nn.Module):
    def __init__(self, modes, width):
        super(Net3d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = FNO3d(modes + 24, modes + 24, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# %%


################################################################
# Loading Data 
################################################################

# %%
#  30055 - 30430 : Initial RBB Camera Data
#  29920 - 29970 : moved RBB Camera Data

if configuration['Case'] == 'RBB Camera':

    data =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rbb_30055_30430.npy')
    data_calib =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rbb_rz_pos_30055_30430.npz')

elif configuration['Case'] == 'RBB Camera - Moved':

    data =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rbb_29920_29970.npy')
    data_calib =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rbb_rz_pos_29920_29970.npz')



# %%
res = configuration['Resolution']
gridx = data_calib['r_pos'][::res, ::res]
gridy = data_calib['z_pos'][::res, ::res]
u_sol = data.astype(np.float32)[:,:,::res, ::res]
# np.random.shuffle(u_sol)


# %%
grid_size_x = u_sol.shape[2]
grid_size_y = u_sol.shape[3]

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

if configuration['Case'] == 'RBB Camera':
    ntrain = 50
    ntest = 9
elif configuration['Case'] == 'RBB Camera - Moved':
    ntrain = 28 
    ntest = 3 


S_x = grid_size_x #Grid Size
S_y = grid_size_y #Grid Size

modes = configuration['Modes']
width = configuration['Width']
# output_size = configuration['Step']

batch_size = configuration['Batch Size']
batch_size2 = batch_size


t1 = default_timer()


T_in = configuration['T_in']
T = configuration['T_out']
T_out = T
# step = configuration['Step']

# gridx = torch.tensor(gridx, dtype=torch.float)
# gridy = torch.tensor(gridy, dtype=torch.float)
# gridx = gridx.reshape(1, S_x, S_y, 1, 1).repeat([1, 1, 1, T, 1])
# gridy = gridx.reshape(1, S_x, S_y, 1, 1).repeat([1, 1, 1, T, 1])

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
# a_normalizer = UnitGaussianNormalizer(train_a)
a_normalizer = RangeNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

# y_normalizer = UnitGaussianNormalizer(train_u)
y_normalizer = RangeNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)

# %%
train_a = train_a.reshape(ntrain,S_x,S_y,1,T_in).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest,S_x,S_y,1,T_in).repeat([1,1,1,T,1])

train_u = train_u.unsqueeze(-1)
test_u = test_u.unsqueeze(-1)

# Using arbitrary R and Z positions sampled uniformly within a specified domain range. 
# pad the location (x,y)

gridx = np.linspace(-1.5, 1.5, 448)[::res]
# x = np.linspace(0, 1, 448)[::res]
gridx = torch.tensor(gridx, dtype=torch.float)
gridx = gridx.reshape(1, S_x, 1, 1, 1).repeat([1, 1, S_y, T, 1])

gridy = np.linspace(-2.0, 2.0, 640)[::res]
# y = np.linspace(0, 1*(640/448), 640)[::res]
gridy = torch.tensor(gridy, dtype=torch.float)
gridy = gridy.reshape(1, 1, S_y, 1, 1).repeat([1, S_x, 1, T, 1])

gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S_x, S_y, 1, 1])

train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1])), dim=-1)

test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1])), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# training and evaluation
################################################################

model = Net3d(modes, width)
wandb.run.summary['Number of Params'] = model.count_params()
print("Number of model params : " + str(model.count_params()))
# model = nn.DataParallel(model, device_ids = [0,1])
model.to(device)
# wandb.watch(model, log='all')

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

# %%

epochs = configuration['Epochs']
y_normalizer.cuda()

# %%
start_time = time.time()
for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        im = model(xx)
        loss = myloss(im.reshape(xx.shape[0], -1), yy.reshape(xx.shape[0], -1))
        train_l2 += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2 = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)


            out = model(xx)
            out = y_normalizer.decode(out)
            loss = myloss(out.reshape(xx.shape[0], -1), yy.reshape(xx.shape[0], -1))
            test_l2 += loss
            
            test_l2 += loss.item()

    t2 = default_timer()
    scheduler.step()
    
    train_loss = train_l2 / ntrain
    test_loss = test_l2 / ntest
    
    print('Epochs: %d, Time: %.2f, Train Loss: %.3e, Test Loss: %.3e' % (ep, t2 - t1, train_loss, test_loss))

    wandb.log({'Train Loss': train_loss, 
               'Test Loss': test_loss})
    
train_time = time.time() - start_time
# %%

torch.save(model.state_dict(), model_loc + '/Models/FNO_rbb_3d_'+run_id + '.pth')
wandb.save(model_loc + '/Models/FNO_rbb_3d_'+run_id + '.pth')

# model = torch.load('NS_FNO_2d_time.pth', map_location=torch.device('cpu'))

# %%

#Testing 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape)
index = 0

with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        xx, yy = xx.to(device), yy.to(device)

        out = model(xx)
        
        out = y_normalizer.decode(out)
        pred_set[index]=out
        index += 1
    
test_l2 = (pred_set - test_u).pow(2).mean()
print('Testing Error: %.3e' % (test_l2))
    
wandb.run.summary['Training Time'] = train_time
wandb.run.summary['Test Error'] = test_l2


# %%


# %%
idx = np.random.randint(0, ntest) 


for idx in range(ntest):
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
    ax.set_ylabel('FNO')

    ax = fig.add_subplot(2,3,5)
    ax.imshow(u_field[:,:,int(T_out/2)], cmap=cm.coolwarm)

    ax = fig.add_subplot(2,3,6)
    ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm)

    wandb.log({"RBB Camera_" +str(idx) : plt})

wandb.run.finish()

