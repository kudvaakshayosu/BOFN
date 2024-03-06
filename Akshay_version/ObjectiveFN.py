#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:38:38 2024
use a test case for simple network gp
@author: kudva.7
"""

from graph_utils import Graph
import torch
from torch import Tensor
from case_studies.covid_simulator import *
from case_studies.Polynomial_simulation import Polynomial
from case_studies.group_testing.src.dynamic_protocol_design import simple_simulation
import copy

torch.set_default_dtype(torch.double)


def function_network_examples(example):

    
        
    if example == 'synthetic_fun1_discrete':
             
        f1 = lambda x: torch.log(x[:,0] + x[:,1])
        f2 = lambda x: 10/(1+x)
    
        
        input_dim = 2
        
        
        def function_network(X: Tensor) -> Tensor:
            """
            Function Network: f1 --> f2 --> f3
            Parameters
            ----------
            X : X[0] is design variable -- Tensor
                X[1] is uncertain variable -- Tensor
        
            Returns
            -------
            Tensor
                Obtains a torch tensor
        
            """
            x_min = torch.tensor([4,-1])
            x_max = torch.tensor([20,1])
            
            X_scale = x_min + (x_max - x_min)*X
            
            #print(X_scale)
            try:
                f0_val = f1(X_scale)
            except:
                f0_val = f1(X_scale.unsqueeze(0))
            f1_val = f2(f0_val) 
    
            return torch.hstack([f0_val,f1_val])
        
        g = Graph(2)
        g.addEdge(0, 1)
        
        active_input_indices = [[0,1],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)
        
        uncertainty_input = [1]
        g.register_uncertainty_variables(uncertain_input_indices=uncertainty_input)
        
        # list of lists
        w1_set = [[0,0.1,0.2,0.7,1]] # Needs to be a list of list
        w_discrete_indices = [1]
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        nominal_w = torch.tensor([[0.7]])
    

    elif example == 'covid_testing':
        """
        In this example, we have two uncertain variables, namely: 
            1) the testing period: [12,13,14,15]: Corresponds to [0,0.33,0.66,1]
            2) The percentage of population infected at t = 0 (prevalence) [0.01,0.012,0.014,0.016,0.018,0.02]: Corresponds to [0,0.2,0.4,0.6,0.8,1]
        """
        
        n_periods = 3
        
        covid_simulator = CovidSimulator(n_periods=n_periods, seed=1)
        input_dim = covid_simulator.input_dim
        n_nodes = covid_simulator.n_nodes

        def function_network(X: Tensor) -> Tensor:        
            # Testing period ranges between 12 and 15 days       
            X[covid_simulator.n_periods] = 12 + X[covid_simulator.n_periods]*3        
            # Prevalence uncertainty ranges between 0.5% to 2%
            X[covid_simulator.n_periods + 1] = 0.01 + X[ covid_simulator.n_periods + 1]*0.01 
                
            return covid_simulator.evaluate(X)
        
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes)    
        
        g.addEdge(0, 3)
        g.addEdge(0, 4)
        g.addEdge(0, 5)
        
        g.addEdge(1, 3)
        g.addEdge(1, 4)
        g.addEdge(1, 5)
        
        g.addEdge(3, 6)
        g.addEdge(3, 7)
        g.addEdge(3, 8)
        
        g.addEdge(4, 6)
        g.addEdge(4, 7)
        g.addEdge(4, 8)
        
        ###########################################################
        # Active input indices
        active_input_indices = []

        for t in range(n_periods):
            for i in range(3):
                active_input_indices.append([t, n_periods, n_periods + 1])
        ##############################################################
        g.register_active_input_indices(active_input_indices)
        
        uncertainty_input = [3,4]
        g.register_uncertainty_variables(uncertainty_input)
        
        # list of lists
        w1_set = [[0,0.33,0.66,1],[0.,0.2,0.4,0.6,0.8,1.]] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        objective_function = lambda Y: -100 * torch.sum(Y[..., [3*t + 2 for t in range(n_periods)]], dim=-1)
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.66, 0.4]])
        
        #g.figure()
        
    elif example == 'polynomial':
        """
        In this example, we have two uncertain variables, namely: 
            1) r: radius
            2) theta: angle
        """
        
       
        
        simulator = Polynomial()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes

        def function_network(X: Tensor) -> Tensor:   
            
            # Design variables
            X[...,0] = torch.tensor(-0.5) + torch.tensor(3.75)*X[...,0]
            X[...,1] = torch.tensor(-0.5) + torch.tensor(4.75)*X[...,1]
            
            # Uncertain variables scale
            X[...,3] = X[...,3]*2*torch.pi #theta
            X[...,2] = X[...,2]*0.5 # r
                
            return simulator.evaluate(X)
        
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes)    
        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[0,2,3],[1,2,3],[0,1,2,3]]

        
        ##############################################################
        g.register_active_input_indices(active_input_indices)
        
        uncertainty_input = [2,3]
        g.register_uncertainty_variables(uncertainty_input)
        
        # list of lists
        w1_set = [[0,0.2,0.4,0.6,1.],[0.0000, 0.100, 0.1250, 0.2000, 0.2500, 0.300, 0.3750, 0.4500, 0.5000, 0.5700, 0.6250, 0.700, 0.7500, 0.8750, 0.9500, 1.0000]] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        objective_function = lambda Y: torch.sum(Y[..., [t for t in range(n_nodes)]], dim=-1)
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.0, 0.0]])
        
        #g.figure()
            

        
        
    
    else:
        print(' Enter a valid test function')
        
    return  function_network, g, nominal_w


    
if __name__ == '__main__':
    
    example_list = ['synthetic_fun1','synthetic_fun1_discrete', 'synthetic_fun2', 'synthetic_fun3', 'covid_testing']
    example = example_list[1]
    
    function_network, g, _ = function_network_examples(example)
    
    test_rest = False
    
    if test_rest: 
        
        uncertainty_input = g.uncertain_input_indices
        design_input = g.design_input_indices    
        
        input_index_info = [design_input, uncertainty_input]
        
        nz = g.nz
        nw = g.nw
        input_dim = g.nx   
        
        #g.figure()
        
        torch.manual_seed(1000)
        x_test =  torch.rand(1,input_dim)  
        
        # Test if everything is okay
        y_true = function_network(x_test)   
        
        
        
        
        # Start the modeling procedure
        Ninit = 30
        n_outs = g.n_nodes
        
        
        
        
        x_init = torch.rand(Ninit, input_dim)
        y_init = torch.zeros(Ninit,n_outs)
        
        for i in range(Ninit):
            y_init[i] = function_network(x_init[i])
            
            
    
   
