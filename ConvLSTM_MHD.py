#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11 March 2022


@author: vgopakum

Convolutional LSTM modelled over the MHD data built using JOREK

Code inspired from this paper : Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
"""
# %%
import wandb
configuration = {"Case": 'MHD',
                "Field": 'rho',
                 "Type": 'ConvLSTM',
                 "Epochs": 1,
                 "Batch Size": 5,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100 ,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh, Sigmoid',
                 "Normalisation Strategy": 'Min-Max. Single',
                 "Batch Normalisation": 'No',
                 "T_in": 20,    
                 "T_out": 50,
                 "Step": 5,
                 "Modes":16,
                 "Width": 32,
                 "Variables":1, 
                 "Noise":0.0, 
                 }

run = wandb.init(project='FNO-Benchmark',
                 notes='',
                 config=configuration,
                 mode='disabled')

run_id = wandb.run.id

wandb.save('FNO_MHD.py')

step = configuration['Step']

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
# model_loc = os.path.dirname(os.path.dirname(os.getcwd()))
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
    def __init__(self, x, low=-1.0, high=1.0):
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

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]).to(device),
                torch.zeros(batch_size, hidden, shape[0], shape[1]).to(device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()

        self.T_in = T_in
        self.step = step
        self.width  = width

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)


    def forward(self, input):
        internal_state = []
        outputs = []


        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)
                

        return outputs, (x, new_c)

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
data =  np.load(data_loc + '/Data/MHD_JOREK_data.npz')

# %%
field = configuration['Field']
if field == 'Phi':
    u_sol = data['Phi'][:,1:,:,:].astype(np.float32)
elif field == 'w':
    u_sol = data['w'][:,1:,:,:].astype(np.float32)
elif field == 'rho':
    u_sol = data['rho'][:,1:,:,:].astype(np.float32)

u_sol = np.nan_to_num(u_sol)
x = data['Rgrid'][0,:].astype(np.float32)
y = data['Zgrid'][:,0].astype(np.float32)
t = data['time'].astype(np.float32)

np.random.shuffle(u_sol)
u = torch.from_numpy(u_sol)
# u = u.permute(0, 2, 3, 1)

ntrain = 100
ntest = 20
S = 100 #Grid Size

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']

batch_size = configuration['Batch Size']
batch_size2 = batch_size


t1 = default_timer()


T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']
################################################################
# load data
################################################################

train_a = u[:ntrain,:T_in,:,:]
train_u = u[:ntrain,T_in:T+T_in,:,:]

test_a = u[-ntest:,:T_in, :, :]
test_u = u[-ntest:,T_in:T+T_in,:,:]

print(train_u.shape)
print(test_u.shape)


# %%
# a_normalizer = RangeNormalizer(train_a)
a_normalizer = MinMax_Normalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

# y_normalizer = RangeNormalizer(train_u)
y_normalizer = MinMax_Normalizer(train_u)
train_u = y_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)
test_u_encoded = y_normalizer.encode(test_u)

# %%

# train_a = train_a.reshape(ntrain,S,S,T_in)
# test_a = test_a.reshape(ntest,S,S,T_in)

# # pad the location (x,y)
# gridx = torch.tensor(x, dtype=torch.float)
# gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
# gridy = torch.tensor(y, dtype=torch.float)
# gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

# # train_a = torch.cat((gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1]), train_a), dim=-1)
# # test_a = torch.cat((gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1]), test_a), dim=-1)

# train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1])), dim=-1)
# test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1])), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# training and evaluation
################################################################


model = ConvLSTM(input_channels=20, hidden_channels=[256, 128, 64, 32, 5], kernel_size=3, step=2,
                    effective_step=[1])##.cuda()
model.to(device)

# wandb.watch(model, log='all')
wandb.run.summary['Number of Params'] = model.count_params()


print("Number of model params : " + str(model.count_params()))

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

# myloss = LpLoss(size_average=False)
myloss = torch.nn.MSELoss()

# gridx = gridx.to(device)
# gridy = gridy.to(device)

# %%

epochs = configuration['Epochs']
# y_normalizer.cuda()

start_time = time.time()
for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        # xx = additive_noise(xx)

        for t in range(0, T, step):
            y = yy[:, t:t + step, : , :]
            im = model(xx)[0][0]
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)

            xx = torch.cat((xx[:, step:, :, :], im), dim=1)

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
                y = yy[:, t:t + step, : , :]
                im = model(xx)[0][0]
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)

                xx = torch.cat((xx[:, step:, :, :], im), dim=1)

            # pred = y_normalizer.decode(pred)
            
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

torch.save(model.state_dict(), model_loc + '/Models/FNO_MHD_' +field+'_'+run_id + '.pth')
wandb.save(model_loc + '/Models/FNO_MHD_' +field+'_'+run_id + '.pth')


# %%

#Testing 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        # xx = additive_noise(xx)
        for t in range(0, T, step):
            y = yy[:, t:t + step, : , :]
            out = model(xx)[0][0]
            loss += myloss(out.reshape(1, -1), y.reshape(1, -1))

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), 1)       
                
            xx = torch.cat((xx[:, step:, :, :], out), dim=1)

        
        # pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
    
MSE_error = (pred_set - test_u_encoded).pow(2).mean()
MAE_error = torch.abs(pred_set - test_u_encoded).mean()
LP_error = loss / (ntest*T/step)

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))
print('(LP) Testing Error: %.3e' % (LP_error))


wandb.run.summary['Training Time'] = train_time
wandb.run.summary['MSE Test Error'] = MSE_error
wandb.run.summary['MAE Test Error'] = MAE_error
wandb.run.summary['LP Test Error'] = LP_error

pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

# %%

idx = np.random.randint(0,ntest) 
u_field = test_u[idx]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
ax.imshow(u_field[0, :,:], cmap=cm.coolwarm)
ax.title.set_text('Initial')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2,3,2)
ax.imshow(u_field[int(T/2),:,:], cmap=cm.coolwarm)
ax.title.set_text('Middle')

ax = fig.add_subplot(2,3,3)
ax.imshow(u_field[-1,:,:], cmap=cm.coolwarm)
ax.title.set_text('Final')

u_field = pred_set[idx]

ax = fig.add_subplot(2,3,4)
ax.imshow(u_field[0,:,:], cmap=cm.coolwarm)
ax.set_ylabel('FNO')

ax = fig.add_subplot(2,3,5)
ax.imshow(u_field[int(T/2),:,:], cmap=cm.coolwarm)

ax = fig.add_subplot(2,3,6)
ax.imshow(u_field[-1,:,:], cmap=cm.coolwarm)


wandb.log({"MHD_" + field: plt})

wandb.run.finish()

# %%
