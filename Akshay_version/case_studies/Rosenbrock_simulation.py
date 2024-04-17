#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:45:30 2024

@author: kudva.7
"""
import torch

class Rosenbrock:
    def __init__(self):
        self.n_nodes = 3
        self.input_dim = 3

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        # 
        Z = X[...,0] + X[...,2]
        W = X[...,1]  
        
        # Network structure
        output[..., 0] = Z**2
        output[..., 1] = (Z - 1)**2
        output[..., 2] = (W - output[..., 0])**2
        
        return output

if __name__ == '__main__':
    
    a = Rosenbrock()
    LB = torch.tensor([-2,0])
    UB = torch.tensor([2, 2])
    
    rand_init = LB + (UB - LB)*torch.rand(20,2)
    
    print(a.evaluate(rand_init))