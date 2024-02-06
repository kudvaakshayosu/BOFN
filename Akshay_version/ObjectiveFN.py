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
from gp_network_utils import GaussianProcessNetwork
import time
from botorch.optim import optimize_acqf
from acquisition_functions import BONSAI_Acquisition, ARBO_UCB
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
torch.set_default_dtype(torch.double)


def function_network_examples(example):

    if example == 'concave_two_dim':
         
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
        
    elif example == 'non_concave_two_dim':
        f1 = lambda x: torch.log(x[:,0] + x[:,1])
        f2 = lambda x: 4.7/(1+x)
        f3 = lambda x: 100*(torch.sin(x[0]) + 2*torch.cos(0.5*x[1]))
        
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
            try:
                f2_val = f3(torch.cat([f1_val,X_scale[1].unsqueeze(0)]))
            except:
                f2_val = f3(torch.cat([f1_val,X_scale[:,1]])).unsqueeze(0)
            
    
            return torch.hstack([f0_val, f1_val, f2_val])
        
        g = Graph(3)
        g.addEdge(0, 1)
        g.addEdge(1, 2)
        
        active_input_indices = [[0,1],[],[1]]
        g.register_active_input_indices(active_input_indices=active_input_indices)
        
        uncertainty_input = [1]
        g.register_uncertainty_variables(uncertain_input_indices=uncertainty_input)
    
    else:
        print(' Enter a valid test function')
        
    return  function_network, g


    
if __name__ == '__main__':
    
    example_list = ['concave_two_dim', 'non_concave_two_dim']
    example = example_list[1]
 
    function_network, g = function_network_examples(example)
    
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
    
    
    
    
    test_stuff = True 
    
    
    
    if test_stuff:
    
    
        # Upper confidence bound
        run_BONSAI = False
        
        if run_BONSAI:
            
            
            model = GaussianProcessNetwork(train_X=x_init, train_Y=y_init, dag=g)
        
        
            test_val = torch.rand(2,input_dim)
        
            post = model.posterior(test_val)
            mean_test, std_test = post.mean_sigma
        
        
            beta = torch.tensor(2)
        
            
            
            ucb_fun = BONSAI_Acquisition(model, beta = torch.tensor(2))
            
            bounds =torch.tensor([[0]*(nz),[1]*(nz)])
            bounds = bounds.type(torch.float)
            
            t1 = time.time()
            z_star, acq_value = optimize_acqf(ucb_fun, bounds, q = 1, num_restarts = 5, raw_samples = 50)
            t2 = time.time()
            print("Time taken for max_{z} min_{w} max_{eta} UCB = ", t2 - t1)
            
            # Lower confidence bound
            lcb_fun = BONSAI_Acquisition(model, beta = torch.tensor(2), maximize = False, fixed_variable = z_star)
            
            bounds =torch.tensor([[0]*(nw),[1]*(nw)])
            bounds = bounds.type(torch.float)
            
            t1 = time.time()
            w_star, acq_value = optimize_acqf(lcb_fun, bounds, q = 1, num_restarts = 5, raw_samples = 10)
            t2 = time.time()
            print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1)
        
        
        else:
            model = SingleTaskGP(x_init, y_init[...,-1].unsqueeze(-1),outcome_transform=Standardize(m=1))
            
            ucb_fun = ARBO_UCB( model, beta = torch.tensor(2), input_indices = input_index_info)
            
            bounds =torch.tensor([[0]*(nz),[1]*(nz)])
            bounds = bounds.type(torch.float)
            
            t1 = time.time()
            z_star, acq_value = optimize_acqf(ucb_fun, bounds, q = 1, num_restarts = 5, raw_samples = 50)
            t2 = time.time()
            print("Time taken for max_{z} min_{w} UCB = ", t2 - t1)
            
            
            lcb_fun = ARBO_UCB( model, beta = torch.tensor(2), input_indices = input_index_info, maximize = False, fixed_variable= z_star)
            
            bounds =torch.tensor([[0]*(nw),[1]*(nw)])
            bounds = bounds.type(torch.float)
            
            t1 = time.time()
            w_star, acq_value = optimize_acqf(lcb_fun, bounds, q = 1, num_restarts = 5, raw_samples = 50)
            t2 = time.time()
            print("Time taken for  min_{w} LCB(w; z_star) = ", t2 - t1)
            
            
        
        
        
        

