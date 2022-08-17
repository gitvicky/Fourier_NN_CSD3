#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:52:51 2022

@author: vgopakum
"""
# %%
import torch 
from torch import nn 
import torch.nn.functional as F


import operator
from functools import reduce
from functools import partial

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

# %%


    
class F_LSTM_Cell(nn.Module):
   def __init__(self, modes, width, batch_first=True):
        super(F_LSTM_Cell, self).__init__()
        
        self.modes = modes
        self.width = width
        
        self.F_ix = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_fx = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_gx = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_io = SpectralConv2d(self.width, self.width, self.modes, self.modes)

        self.F_hi = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_hf = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_hg = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.F_ho = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        
        self.W_ix = nn.Conv1d(self.width, self.width, 1)
        self.W_fx = nn.Conv1d(self.width, self.width, 1)
        self.W_gx = nn.Conv1d(self.width, self.width, 1)
        self.W_io = nn.Conv1d(self.width, self.width, 1)

        self.W_hi = nn.Conv1d(self.width, self.width, 1)
        self.W_hf = nn.Conv1d(self.width, self.width, 1)
        self.W_hg = nn.Conv1d(self.width, self.width, 1)
        self.W_ho = nn.Conv1d(self.width, self.width, 1)

        self.bn_0 = torch.nn.BatchNorm2d(self.width)
        self.bn_1 = torch.nn.BatchNorm2d(self.width)
        self.bn_2 = torch.nn.BatchNorm2d(self.width)
        self.bn_3 = torch.nn.BatchNorm2d(self.width)
        self.bn_4 = torch.nn.BatchNorm2d(self.width)
        self.bn_5 = torch.nn.BatchNorm2d(self.width)
        self.bn_6 = torch.nn.BatchNorm2d(self.width)
        self.bn_7 = torch.nn.BatchNorm2d(self.width)
        
   def forward(self, x, h, c):
       
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

#Input Gate 
        a = x.permute(0, 3, 2, 1)
        a1 = self.F_ix(a)
        a2 = self.W_ix(a.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        a = self.bn_0(a1+a2)
        a = F.relu(a)
        a = a.permute(0, 2, 3, 1)

        b = h.permute(0, 3, 2, 1)
        b1 = self.F_hi(b)
        b2 = self.W_hi(b.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        b = self.bn_1(b1+b2)
        b = F.relu(b)
        b = b.permute(0, 2, 3, 1)

        i = torch.sigmoid(a + b)

#Forget Gate
        a = x.permute(0, 3, 2, 1)
        a1 = self.F_fx(a)
        a2 = self.W_fx(a.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        a = self.bn_2(a1+a2)
        a = F.relu(a)
        a = a.permute(0, 2, 3, 1)

        b = h.permute(0, 3, 2, 1)
        b1 = self.F_hf(b)
        b2 = self.W_hf(b.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        b = self.bn_3(b1+b2)
        b = F.relu(b)
        b = b.permute(0, 2, 3, 1)

        f = torch.sigmoid(a + b)

# G 

        a = x.permute(0, 3, 2, 1)
        a1 = self.F_gx(a)
        a2 = self.W_gx(a.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        a = self.bn_4(a1+a2)
        a = F.relu(a)
        a = a.permute(0, 2, 3, 1)

        b = h.permute(0, 3, 2, 1)
        b1 = self.F_hg(b)
        b2 = self.W_hg(b.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        b = self.bn_5(b1+b2)
        b = F.relu(b)
        b = b.permute(0, 2, 3, 1)

        g = torch.tanh(a + b)


#Output Gate 


        a = x.permute(0, 3, 2, 1)
        a1 = self.F_io(a)
        a2 = self.W_io(a.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        a = self.bn_6(a1+a2)
        a = F.relu(a)
        a = a.permute(0, 2, 3, 1)

        b = h.permute(0, 3, 2, 1)
        b1 = self.F_ho(b)
        b2 = self.W_ho(b.contiguous().view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        b = self.bn_7(b1+b2)
        b = F.relu(b)
        b = b.permute(0, 2, 3, 1)

        o = torch.sigmoid(a + b)   

#Cell State
        c =  f*c + i*g

#Hidden State and Output y 
        h = o * torch.tanh(c)
        y = h
        
        return y, h.clone().detach(), c.clone().detach()
    
   def count_params(self):
        count = 0
        for p in self.parameters():
            count += reduce(operator.mul, list(p.size()))
    
        return count


class F_LSTM(nn.Module):
   def __init__(self, modes, width, n_output, n_hidden, n_cellsize, n_cells, T_in, batch_first=True):
        super(F_LSTM, self).__init__()
        
        self.modes = modes
        self.width = width
        self.n_output = n_output 
        self.n_hidden = n_hidden
        self.n_cellsize = n_cellsize
        
        self.linear_in_x = nn.Linear(T_in+2, self.width)
        self.linear_in_h = nn.Linear(self.n_hidden, self.width)
        self.linear_in_c = nn.Linear(self.n_cellsize, self.width)
        
        self.linear_out_x = nn.Linear(self.width, self.n_output)
        self.linear_out_h = nn.Linear(self.width , self.n_hidden)
        self.linear_out_c = nn.Linear(self.width , self.n_cellsize)

        self.F_LSTM_Cells = nn.ModuleList()
        
        for ii in range(n_cells):
            self.F_LSTM_Cells.append(F_LSTM_Cell(self.modes, self.width))

        
   def forward(self, x, h, c):
       
        x = self.linear_in_x(x) 
        h = self.linear_in_h(h)
        c = self.linear_in_c(c)

        for cell in self.F_LSTM_Cells:
            x, h, c = cell(x, h, c)

        y = self.linear_out_x(x)
        h = self.linear_out_h(h)
        c = self.linear_out_h(c)

        return y, h.clone().detach(), c.clone().detach()
    
   def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
    
        return c

    
# %%
modes = 16
width = 32 
hidden_size = 32
output_size = 10
cell_size = 32
num_cells = 1
T_in = 20 

model = F_LSTM(modes, width, output_size, hidden_size, cell_size, num_cells, T_in)

# %%


inps = torch.ones((10, 99, 99, 22)) #Batch Size, gridx, gridy, T_in + 2
hidden = torch.ones((10, 99, 99, 32)) #Batch Size, gridx, gridy, hidden size
cell_state = torch.ones((10, 99, 99, 32)) #Batch Size, gridx, gridy, cell size


y, h, c = model(inps, hidden, cell_state)
# %%
