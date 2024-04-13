#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:22:25 2024

@author: kudva.7
"""


import torch

class Sine:
    def __init__(self):
        self.n_nodes = 6
        self.input_dim = 4

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))         
        
        output[..., 0] = X[...,0] + X[...,2]
        output[..., 1] = -1*torch.sin(2*torch.pi*output[...,0]**2)
        output[..., 2] = -1*(output[...,0]**2 + 0.2*output[...,0])
        output[..., 3] =  X[...,1] + X[...,3]
        output[..., 4] = -1*torch.sin(2*torch.pi*output[...,3]**2)
        output[..., 5] = -1*(output[...,3]**2 + 0.2*output[...,3])
        
        return output
    
    
if __name__ == '__main__':
    
    a = Sine()
    LB = torch.tensor([-1.,-1.,-0.5,-0.5])
    UB = torch.tensor([1.,1.,0.5,0.5])
    
    rand_init = LB + (UB - LB)*torch.rand(20,4)
    
    print(a.evaluate(rand_init))