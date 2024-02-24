#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:28:50 2024

@author: kudva.7
"""
from case_studies.Dropwave_simulation import Dropwave
from case_studies.generate_TS import generate_gaussian_nodes
import torch
from torch import Tensor
from botorch.acquisition.objective import GenericMCObjective
from graph_utils import Graph
import sys

torch.set_default_dtype(torch.float64)

example_list = ['dropwave', 'config_one', 'config_two', 'config_three']


def function_network_examples(example):

    if example == 'dropwave':
    
        dropwave = Dropwave()
        input_dim = dropwave.input_dim
        n_nodes = 2
        problem = 'dropwave'
        
        def function_network(X: Tensor):
            return dropwave.evaluate(X=X)
        
        # Underlying DAG
        g = Graph(2)
        g.addEdge(0, 1)
         
        active_input_indices = [[0,1],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)
        
        
        # Function that maps the network output to the objective value
        network_to_objective_transform = lambda Y: Y[..., -1] + 1e-1*torch.randn(1)        
        g.define_objective(network_to_objective_transform) 
        
        
    elif example == 'config_one':
        F11, F12, F13, F14, F21, F22, F23 = generate_gaussian_nodes(noise = 1e-2)

        def function_network(X: Tensor):
            try:
                y1 = F11(X[0].unsqueeze(0))
                y2 = F21(torch.cat((y1, X[1].unsqueeze(0))).unsqueeze(0))
                y3 = F13(y2)
            except:
                try:
                    X = X.squeeze(0)
                    y1 = F11(X[0].unsqueeze(0))
                except:
                    X = X.squeeze(0).squeeze(-1)
                    y1 = F11(X[0].unsqueeze(0))
                y2 = F21(torch.cat((y1, X[1].unsqueeze(0))).unsqueeze(0))
                y3 = F13(y2)
            return torch.cat((y1,y2,y3))
        
        
        # Underlying DAG
        g = Graph(3)
        g.addEdge(0, 1)
        g.addEdge(1, 2)
        
        active_input_indices = [[0],[1],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)
        
        #Setting the default model hyperparameters
        g.set_model_hyperparameters(noise_level = 1e-2)
        g.figure()
        
    elif example == 'config_two':
        F11, F12, F13, F14, F21, F22, F23 = generate_gaussian_nodes()

        def function_network(X: Tensor):
            try:
                y1 = F11(X[0].unsqueeze(0))
                y2 = F12(X[1].unsqueeze(0))
                
                y3 = F21(torch.cat((y1, y2)).unsqueeze(0))
                y4 = F22(torch.cat((y1, y2)).unsqueeze(0))
                
                y5 = F23(torch.cat((y3, y4)).unsqueeze(0))
            except:
                try:
                    X = X.squeeze(0)
                    y1 = F11(X[0].unsqueeze(0))
                    y2 = F12(X[1].unsqueeze(0))
                except:
                    X = X.squeeze(0).squeeze(-1)
                    y1 = F11(X[0].unsqueeze(0))
                    y2 = F12(X[1].unsqueeze(0))
                
                
                y3 = F21(torch.cat((y1, y2)).unsqueeze(0))
                y4 = F22(torch.cat((y1, y2)).unsqueeze(0))
                
                y5 = F23(torch.cat((y3, y4)).unsqueeze(0))
            
            return torch.cat((y1,y2,y3,y4,y5))
        
        
        # Underlying DAG
        g = Graph(5)
        g.addEdge(0, 2)
        g.addEdge(1, 2)
        g.addEdge(0, 3)
        g.addEdge(1, 3)
        g.addEdge(2, 4)
        g.addEdge(3, 4)
        
        active_input_indices = [[0],[1],[],[],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)  
        
        # Setting the default model hyperparameters
        g.set_model_hyperparameters()
        
        g.figure()
        
    elif example == 'config_three':
        F11, F12, F13, F14, F21, F22, F23 = generate_gaussian_nodes()

        def function_network(X: Tensor):
            try:
                y1 = F21(X.unsqueeze(0))   
            except:
                try:
                    y1 = F21(X.unsqueeze(0))   
                except:
                    X = X.squeeze(-1).squeeze(-2)
                    y1 = F21(X.unsqueeze(0)) 
            
            y2 = F11(y1)
            
            y3 = F12(y1)
            
            y4 = F13(y1)
            
            y5 = F14(y1)
                
            try:
                out = torch.cat((y1,y2,y3,y4,y5))
            except:
                out = torch.cat((y1.squeeze(0),y2,y3,y4,y5))
            
            return out
        
        
        # Underlying DAG
        g = Graph(5)
        g.addEdge(0, 1)
        g.addEdge(0, 2)
        g.addEdge(0, 3)
        g.addEdge(0, 4)
        
        active_input_indices = [[0,1],[],[],[],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)  
        
        network_to_objective_transform = lambda Y: torch.sum(Y[..., [t for t in range(1,g.n_nodes)]], dim=-1)
        g.define_objective(network_to_objective_transform) 
        
        # Setting the default model hyperparameters
        g.set_model_hyperparameters()
        
        g.figure()
        
        
        
    else:
        print('Please enter a valid example problem')
        sys.exit()
        
    return function_network, g







if __name__ == '__main__':
    
    print('Testing testing...')
    a,b =  function_network_examples('config_one')
    

    
