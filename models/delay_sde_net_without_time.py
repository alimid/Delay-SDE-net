#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:42:11 2019

@author: alimid
"""

import torch
import torch.nn as nn


__all__ = ['SDENet_drift','SDENet_diffusion','SDENet_epistemic']

class Drift(nn.Module):
    def __init__(self,p,m):
        super(Drift, self).__init__()
        self.fc = nn.Linear(m*(p+1),2*(m*(p+1)))
        self.fc2 = nn.Linear(2*(m*(p+1)),m, bias=True) #m
        self.softplus = nn.Softplus(500)
    def forward(self, x):
        out = self.softplus(self.fc(x))
        out = self.fc2(out)
        return out    

class Diffusion(nn.Module):
    def __init__(self,p,m):
        super(Diffusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(m*(p+1), int(2*(m*(p+1))))
        self.fc2 = nn.Linear(int(2*(m*(p+1))), 1, bias=True)
        self.softplus = nn.Softplus(500)
    def forward(self, x):
        out_diff = self.softplus(self.fc1(x))
        out_diff = self.softplus(self.fc2(out_diff))
        return out_diff
    
class Diffusion_epistemic(nn.Module):
    def __init__(self,p,m):
        super(Diffusion_epistemic, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(m*(p+1)+1, int(2*(m*(p+1)+1)))
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(int(2*(m*(p+1)+1)), 1, bias=True)
        nn.init.kaiming_uniform_(self.fc2.weight)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(500) #100
    def forward(self, x):
        out = self.softplus(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out
    
    
class SDENet_drift(nn.Module):
    def __init__(self, m,p,l,incl_y=True):
        super(SDENet_drift, self).__init__()
        self.drift = Drift(p,m)
        self.deltat = 1
        self.p = p
        self.m = m
        self.layer_depth = 1
        self.incl_y = incl_y
    def forward(self, x, t=None):
        inputs = x
        out = inputs[self.p:]
        out_last = inputs[:len(inputs)-self.p]
        out_lags = out_last.clone()
        
        for i in range(1,self.p+1):
            out_lag = inputs[i:len(inputs)-self.p+i]
            out_lags = torch.column_stack((out_lags, out_lag))

        mask = torch.arange(self.m,len(out_lags[0])-1)
        
        for i in range(self.layer_depth):
            if self.incl_y == True:
                out = out + self.drift(out_lags)*self.deltat
            else:
                out = self.drift(out_lags)*self.deltat
            drift_out = self.drift(out_lags)
            out_lags = torch.index_select(out_lags,1,mask)
            out_lags = torch.column_stack((out_lags,out))
        
        mean = out[:,0]
        return mean, drift_out 

        
class SDENet_epistemic(nn.Module):
    def __init__(self, m,p):
        super(SDENet_epistemic, self).__init__()
        self.deltat = 1
        self.sigma = 1
        self.p = p
        self.diffusion_epistemic = Diffusion_epistemic(p,m)
    def forward(self, x):
        inputs = x
        out = inputs

        final_out = self.diffusion_epistemic(out.detach().float()) 
        return final_out
    
        
        
class SDENet_diffusion(nn.Module):
    def __init__(self, m,p):
        super(SDENet_diffusion, self).__init__()
        self.diffusion = Diffusion(p,m)
        self.deltat = 1
        self.sigma = 1
        self.p = p
    def forward(self, x, t=None):
        inputs = x
        out = inputs[:len(inputs)-self.p]
        out_lags = out.clone()
        for i in range(1,self.p+1):
            out_lag = inputs[i:len(inputs)-self.p+i]
            out_lags = torch.column_stack((out_lags, out_lag))
            

        final_out = self.diffusion(out_lags)*self.deltat  
        final_out = final_out[:len(final_out)]
        return final_out
    


