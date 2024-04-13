#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:01:00 2024

@author: kudva.7
"""

import torch
from torch import Tensor
#import matplotlib.pyplot as plt

class Cliff:
    def __init__(self):
        self.n_nodes = 5
        self.input_dim = 10

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        X1 = X[...,0] + 0.5*torch.sin(X[...,5])
        X2 = X[...,1] + 0.5*torch.sin(X[...,6])
        X3 = X[...,2] + 0.5*torch.sin(X[...,7])
        X4 = X[...,3] + 0.5*torch.sin(X[...,8])
        X5 = X[...,4] + 0.5*torch.sin(X[...,9])
        
        
        
        output[..., 0] = -10/(1 + 0.3*torch.exp(6*X1)) - 0.2*X1**2
        output[..., 1] = -10/(1 + 0.3*torch.exp(6*X2)) - 0.2*X2**2
        output[..., 2] = -10/(1 + 0.3*torch.exp(6*X3)) - 0.2*X3**2
        output[..., 3] = -10/(1 + 0.3*torch.exp(6*X4)) - 0.2*X4**2
        output[..., 4] = -10/(1 + 0.3*torch.exp(6*X5)) - 0.2*X5**2
        
        return output
    
    
if __name__ == '__main__':
    
    a = Cliff()
    LB = torch.tensor([0.,0.,0.,0.,0.,-torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2])
    UB = torch.tensor([5.,5.,5.,5.,5.,torch.pi/2,torch.pi/2,torch.pi/2,torch.pi/2,torch.pi/2])
    
    rand_init = LB + (UB - LB)*torch.rand(20,10)
    
    print(a.evaluate(rand_init))
    
    
    
    
    
    
        