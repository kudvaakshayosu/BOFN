#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:28:50 2024

@author: kudva.7
"""
from Dropwave_simulation import Dropwave
import torch
from torch import Tensor
from botorch.acquisition.objective import GenericMCObjective
from graph_utils import Graph
import sys

torch.set_default_dtype(torch.float64)

example_list = ['dropwave', 'toy_problem']


def function_network_examples(example):

    if example == 'dropwave':
    
        dropwave = Dropwave()
        input_dim = dropwave.input_dim
        n_nodes = 2
        
        def function_network(X: Tensor):
            return dropwave.evaluate(X=X)
        
        # Underlying DAG
        g = Graph(2)
        g.addEdge(0, 1)
         
        active_input_indices = [[0,1],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)
        
        
        # Function that maps the network output to the objective value
        network_to_objective_transform = lambda Y: Y[..., -1]         
        g.define_objective(network_to_objective_transform) 
                
    else:
        print('Please enter a valid example problem')
        sys.exit()
        
    return function_network, g


if __name__ == '__main__':
    
    print('Testing testing...')
    a,b =  function_network_examples(example_list[0]) 
