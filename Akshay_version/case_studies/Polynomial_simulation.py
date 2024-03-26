#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:52:07 2024

@author: kudva.7
"""

import torch
from torch import Tensor

class Polynomial:
    def __init__(self):
        self.n_nodes = 3
        self.input_dim = 4

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        X1 = X[...,0] + X[...,2]*torch.cos(X[...,3])
        X2 = X[...,1] + X[...,2]*torch.sin(X[...,3])
        
        output[..., 0] = -2*(X1)**6 + 12.2*(X1)**5 - 21.2*(X1)**4 - 6.2*(X1) + 6.4*(X1)**3 + 4.7*(X1)**2 
        output[..., 1] = -1*(X2)**6 + 11*(X2)**5 - 43.3*(X2)**4 + 10*(X2) + 74.8*(X2)**3 - 56.9*(X2)**2
        output[..., 2] = + 4.1*(X1)*(X2) + 0.1*((X2)**2)*((X1)**2) - 0.4*((X2)**2)*(X1) - 0.4*((X1)**2)*(X2)
        
        return output




if __name__ == '__main__':
    """        Hello world!
    """
    
    torch.set_default_dtype(torch.float64)
    dropwave = Polynomial()
    input_dim = dropwave.input_dim

    fun = lambda z: (-2*(z[:,0])**6 + 12.2*(z[:,0])**5 - 21.2*(z[:,0])**4 - 6.2*(z[:,0]) + 6.4*(z[:,0])**3 + 4.7*(z[:,0])**2 
    - (z[:,1])**6 + 11*(z[:,1])**5 - 43.3*(z[:,1])**4 + 10*(z[:,1]) + 74.8*(z[:,1])**3 - 56.9*(z[:,1])**2
    + 4.1*(z[:,0])*(z[:,1]) + 0.1*((z[:,1])**2)*((z[:,0])**2) - 0.4*((z[:,1])**2)*(z[:,0]) - 0.4*((z[:,0])**2)*(z[:,1]))

    def function_network(X: Tensor):
        return dropwave.evaluate(X=X).sum(dim = -1)
    
    # Sanity check:
    a = function_network(torch.tensor([[0.1, 0.1 ,0.5, torch.pi]]))
    b = fun(torch.tensor([[0.1 - 0.5 ,0.1]]))
    
    print(a - b)
