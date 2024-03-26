#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:32:11 2024

@author: kudva.7
"""
import torch
from torch import Tensor

class Dropwave:
    def __init__(self):
        self.n_nodes = 2
        self.input_dim = 2

    def evaluate(self, X):
        X_scaled = 10.24 * X - 5.12
        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))
        norm_X = torch.norm(X_scaled, dim=-1)
        output[..., 0] = norm_X
        output[..., 1] = (1.0 + torch.cos(12.0 * norm_X)) /(2.0 + 0.5 * (norm_X ** 2))
        return output


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    dropwave = Dropwave()
    input_dim = dropwave.input_dim
    n_nodes = 2
    problem = 'dropwave'

    def function_network(X: Tensor):
        return dropwave.evaluate(X=X)
    
    # Sanity check:
    function_network(torch.tensor([0.5,0.5]))
