#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:34:57 2022


@author: vgopakum

FNO 2d time on RBB Camera Data 


"""

# %%
import wandb
configuration = {"Case": 'RBB Camera - Mixed',
                 "Calibration": 'Calcam',
                 "Epochs": 100,
                 "Batch Size": 2,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 20 ,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'ReLU',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10, 
                 "T_out": 50,
                 "Step": 1,
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

wandb.save('FNO_rbb_mixed.py')


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

# additive_noise = AddGaussianNoise(0.0, configuration['Noise'])
# additive_noise.cuda()

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
'''
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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)


        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)


        #Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))

        return x
'''
class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(T_in+2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, step)
  
    def forward(self, x):
      batchsize = x.shape[0]
      size_x, size_y = x.shape[1], x.shape[2]

      x = self.fc0(x)
      x = x.permute(0, 3, 1, 2)

      x1 = self.conv0(x)
      x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
      x = self.bn0(x1 + x2)
      x = F.relu(x)
      x1 = self.conv1(x)
      x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
      x = self.bn1(x1 + x2)
      x = F.relu(x)
      x1 = self.conv2(x)
      x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
      x = self.bn2(x1 + x2)
      x = F.relu(x)
      x1 = self.conv3(x)
      x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
      x = self.bn3(x1 + x2)


      x = x.permute(0, 2, 3, 1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width)


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


data_1 =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rbb_30055_30430.npy')
data_calib_1 =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rbb_rz_pos_30055_30430.npz')

data_2 =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rbb_29920_29970.npy')
data_calib_2 =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rbb_rz_pos_29920_29970.npz')

# data =  np.load(data_loc + '/Data/Cam_Data/rbb_fno_data.npy')
# data_calib =  np.load(data_loc + '/Data/Cam_Data/rbb_rz_pos.npz')

# %%
#
res = configuration['Resolution']
u_sol_1 = data_1.astype(np.float32)[:,:,::res, ::res]

grid_size_x = u_sol_1.shape[2]
grid_size_y = u_sol_1.shape[3]

u_1 = torch.from_numpy(u_sol_1)
u_1 = u_1.permute(0, 2, 3, 1)

u_sol_2 = data_2.astype(np.float32)[:,:,::res, ::res]
u_2 = torch.from_numpy(u_sol_2)
u_2 = u_2.permute(0, 2, 3, 1)

ntrain_1 = 50
ntest_1 = 9
ntrain_2 = 28 
ntest_2 = 3 


S_x = grid_size_x #Grid Size
S_y = grid_size_y #Grid Size

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']

batch_size = configuration['Batch Size']
batch_size2 = batch_size


t1 = default_timer()


T_in = configuration['T_in']
T = configuration['T_out']
T_out = T
step = configuration['Step']
################################################################
# load data
################################################################

train_a_1 = u_1[:ntrain_1,:,:,:T_in]
train_u_1 = u_1[:ntrain_1,:,:,T_in:T+T_in]

test_a_1 = u_1[-ntest_1:,:,:,:T_in]
test_u_1 = u_1[-ntest_1:,:,:,T_in:T+T_in]

train_a_2 = u_2[:ntrain_2,:,:,:T_in]
train_u_2 = u_2[:ntrain_2,:,:,T_in:T+T_in]

test_a_2 = u_2[-ntest_2:,:,:,:T_in]
test_u_2 = u_2[-ntest_2:,:,:,T_in:T+T_in]


# %%
# a_normalizer = UnitGaussianNormalizer(train_a)
a_normalizer = RangeNormalizer(torch.vstack((train_a_1, train_a_2)))
train_a_1 = a_normalizer.encode(train_a_1)
test_a_1 = a_normalizer.encode(test_a_1)
train_a_2 = a_normalizer.encode(train_a_2)
test_a_2 = a_normalizer.encode(test_a_2)

# y_normalizer = UnitGaussianNormalizer(train_u)
y_normalizer = RangeNormalizer(torch.vstack((train_u_1, train_u_2)))
train_u_1 = y_normalizer.encode(train_u_1)
train_u_2 = y_normalizer.encode(train_u_2)

# test_u = y_normalizer.encode(test_u)

# %%


#Using arbitrary R and Z positions sampled uniformly within a specified domain range. 
# pad the location (x,y)
# x = np.linspace(-1.5, 1.5, 448)[::res]
# gridx = torch.tensor(x, dtype=torch.float)
# gridx = gridx.reshape(1, S_x, 1, 1).repeat([1, 1, S_y, 1])

# y = np.linspace(-2.0, 2.0, 640)[::res]
# gridy = torch.tensor(y, dtype=torch.float)
# gridy = gridy.reshape(1, 1, S_y, 1).repeat([1, S_x, 1, 1])

# gridx_1 = gridx
# gridx_2 = gridx
# gridy_1 = gridy
# gridy_2 = gridy


#Using the calibrated R and Z positions averaged over the time and shots. 
# gridx_1 = data_calib_1['r_pos'][::res, ::res]
# gridy_1 = data_calib_1['z_pos'][::res, ::res]
# gridx_1 = torch.tensor(gridx_1, dtype=torch.float)
# gridy_1 = torch.tensor(gridy_1, dtype=torch.float)
# gridx_1 = gridx_1.reshape(1, S_x, S_y, 1)
# gridy_1 = gridy_1.reshape(1, S_x, S_y, 1)


train_a_1 = torch.cat((train_a_1, gridx_1.repeat([ntrain_1,1,1,1]), gridy_1.repeat([ntrain_1,1,1,1])), dim=-1)
test_a_1 = torch.cat((test_a_1, gridx_1.repeat([ntest_1,1,1,1]), gridy_1.repeat([ntest_1,1,1,1])), dim=-1)

gridx_1 = gridx_1.to(device)
gridy_1 = gridy_1.to(device)

train_loader_1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a_1, train_u_1), batch_size=batch_size, shuffle=True)
test_loader_1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a_1, test_u_1), batch_size=batch_size, shuffle=False)

#Using the calibrated R and Z positions averaged over the time and shots. 
gridx_2 = data_calib_2['r_pos'][::res, ::res]
gridy_2 = data_calib_2['z_pos'][::res, ::res]
gridx_2 = torch.tensor(gridx_2, dtype=torch.float)
gridy_2 = torch.tensor(gridy_2, dtype=torch.float)
gridx_2 = gridx_2.reshape(1, S_x, S_y, 1)
gridy_2 = gridy_2.reshape(1, S_x, S_y, 1)


train_a_2 = torch.cat((train_a_2, gridx_2.repeat([ntrain_2,1,1,1]), gridy_2.repeat([ntrain_2,1,1,1])), dim=-1)
test_a_2 = torch.cat((test_a_2, gridx_2.repeat([ntest_2,1,1,1]), gridy_2.repeat([ntest_2,1,1,1])), dim=-1)

gridx_2 = gridx_2.to(device)
gridy_2 = gridy_2.to(device)

train_loader_2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a_2, train_u_2), batch_size=batch_size, shuffle=True)
test_loader_2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a_2, test_u_2), batch_size=batch_size, shuffle=False)


t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# training and evaluation
################################################################

model = Net2d(modes, width)
wandb.run.summary['Number of Params'] = model.count_params()
print("Number of model params : " + str(model.count_params()))

# model = nn.DataParallel(model, device_ids = [0,1])
model.to(device)

# wandb.watch(model, log='all')



optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

myloss = LpLoss(size_average=False)


# %%

epochs = configuration['Epochs']
y_normalizer.cuda()

start_time = time.time()
for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0

    data_index = np.random.randint(1,3)
    if data_index == 1:
        train_loader = train_loader_1
        test_loader = test_loader_1
        gridx = gridx_1
        gridy = gridy_1
        ntrain = ntrain_1
        ntest = ntest_1

    if data_index == 2:
        train_loader = train_loader_2
        test_loader = test_loader_2
        gridx = gridx_2
        gridy = gridy_2
        ntrain = ntrain_2
        ntest = ntest_2

    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        # xx = additive_noise(xx)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                
                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([xx.shape[0], 1, 1, 1]), gridy.repeat([xx.shape[0], 1, 1, 1])), dim=-1)

            pred = y_normalizer.decode(pred)
            
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    
    train_loss = train_l2_full / ntrain
    test_loss = test_l2_full / ntest
    
    print('Epochs: %d, Time: %.2f, Train Loss per step: %.3e, Train Loss: %.3e, Test Loss per step: %.3e, Test Loss: %.3e' % (ep, t2 - t1, train_l2_step / ntrain / (T / step), train_loss, test_l2_step / ntest / (T / step), test_loss))

    wandb.log({'Train Loss': train_loss, 
               'Test Loss': test_loss})
    
train_time = time.time() - start_time
# %%

torch.save(model.state_dict(), model_loc + '/Models/FNO_rbb_'+run_id + '.pth')
wandb.save(model_loc + '/Models/FNO_rbb_'+run_id + '.pth')

# model = torch.load('NS_FNO_2d_time.pth', map_location=torch.device('cpu'))

# %%

#Testing 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a_1, test_u_1), batch_size=1, shuffle=False)
gridx = gridx_1
gridy = gridy_1
pred_set = torch.zeros(test_u_1.shape)
index = 0

with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        xx, yy = xx.to(device), yy.to(device)
        # xx = additive_noise(xx)
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            out = model(xx)
            
            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       
                
            xx = torch.cat((xx[..., step:-2], out,
                                gridx.repeat([1, 1, 1, 1]), gridy.repeat([1, 1, 1, 1])), dim=-1)
        
        pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
    
test_l2 = (pred_set - test_u_1).pow(2).mean()
print('Testing Error: %.3e' % (test_l2))
    
wandb.run.summary['Test Error - 1'] = test_l2


# %%
idx = np.random.randint(0, ntest_1) 
u_field = test_u_1[idx].cpu().detach().numpy()

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

wandb.log({"RBB Camera - 1 " : plt})


# %%

#Testing 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a_2, test_u_2), batch_size=1, shuffle=False)
gridx = gridx_2
gridy = gridy_2
pred_set = torch.zeros(test_u_2.shape)
index = 0

with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        xx, yy = xx.to(device), yy.to(device)
        # xx = additive_noise(xx)
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            out = model(xx)
            
            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       
                
            xx = torch.cat((xx[..., step:-2], out,
                                gridx.repeat([1, 1, 1, 1]), gridy.repeat([1, 1, 1, 1])), dim=-1)
        
        pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
    
test_l2 = (pred_set - test_u_2).pow(2).mean()
print('Testing Error: %.3e' % (test_l2))
    
wandb.run.summary['Test Error - 2'] = test_l2


# %%
idx = np.random.randint(0, ntest_2)   
u_field = test_u_2[idx].cpu().detach().numpy()

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

wandb.log({"RBB Camera - 2 " : plt})

# %%
wandb.run.summary['Training Time'] = train_time
wandb.run.finish()

