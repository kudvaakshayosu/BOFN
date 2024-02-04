#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:38:38 2024
use a test case for simple network gp
@author: kudva.7
"""

from graph_utils import Graph
from botorch.acquisition.objective import GenericMCObjective, ScalarizedPosteriorTransform
import torch
from torch import Tensor
from gp_network_utils import GaussianProcessNetwork
import time

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from acquisition_functions import BONSAI_Acquisition 

torch.set_default_dtype(torch.double)

example_number = 1

if example_number == 1:
    
    f1 = lambda x: -2*x**2 + 12.2*x**5 - 21.2*x**4 - 6.2*x + 6.4*x**3 + 4.7*x**2
    f2 = lambda x: -x**6 + 11*x**5 -43.3*x**4 + 10*x + 74.8*x**3 - 56.9*x**2
    f3 = lambda x,y: 4.1*x*y + 0.1*(y**2)*(x**2) - 0.4*(y**2)*x - 0.4*(x**2)*y
    
    input_dim = 2
    
    
    def function_network(X: Tensor) -> Tensor:
        """
        Function Network: f1 --> f2 --> f3
        Parameters
        ----------
        X : Tensor
            DESCRIPTION.
    
        Returns
        -------
        Tensor
            Obtains a torch tensor
    
        """
        x_min = torch.tensor([-1,-1])
        x_max = torch.tensor([4,4])
        
        X_scale = x_min + (x_max - x_min)*X
        
        #print(X_scale)
        
        try:
            f0_val = f1(X_scale[:,0])
            f1_val = f2(X_scale[:,1]) + f0_val
            f2_val = f3(X_scale[:,0],X[:,1]) + f1_val
            out = torch.stack([f0_val,f1_val,f2_val], dim = 1)
            
        except:
            f0_val = f1(X_scale[0])
            f1_val = f2(X_scale[1]) + f0_val
            f2_val = f3(X_scale[0],X[1]) + f1_val 
            out = torch.tensor([f0_val,f1_val,f2_val])
        
        return out
    
    g = Graph(3)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    active_input_indices = [[0],[1],[0,1]]
    g.figure()
    
else:
     
    f1 = lambda x: torch.log(x[:,0] + x[:,1])
    f2 = lambda x: 10/(1+x)

    
    input_dim = 2
    
    
    def function_network(X: Tensor) -> Tensor:
        """
        Function Network: f1 --> f2 --> f3
        Parameters
        ----------
        X : Tensor
            DESCRIPTION.
    
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

        return f1_val
    
    g = Graph(2)
    g.addEdge(0, 1)
    
    active_input_indices = [[0,1],[]]
    g.register_active_input_indices(active_input_indices=active_input_indices)
    
    uncertainty_input = [1]
    g.register_uncertainty_variables(uncertain_input_indices=uncertainty_input)
    
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


model = GaussianProcessNetwork(train_X=x_init, train_Y=y_init, dag=g)


test_val = torch.rand(2,input_dim)

post = model.posterior(test_val)
mean_test, std_test = post.mean_sigma


beta = torch.tensor(2)

nz = model.dag.nz
nw = model.dag.nw
nx = model.dag.nx



# Upper confidence bound
ucb_fun = BONSAI_Acquisition(model, beta = torch.tensor(2))

bounds =torch.tensor([[0]*(nz),[1]*(nz)])
bounds = bounds.type(torch.float)

t1 = time.time()
z_star, acq_value = optimize_acqf(ucb_fun, bounds, q = 1, num_restarts = 5, raw_samples = 500)
t2 = time.time()
print("Time taken for max_{z} min_{w} max_{eta} UCB = ", t2 - t1)

# Lower confidence bound
lcb_fun = BONSAI_Acquisition(model, beta = torch.tensor(2), maximize = False, fixed_variable = z_star)

bounds =torch.tensor([[0]*(nw),[1]*(nw)])
bounds = bounds.type(torch.float)

t1 = time.time()
w_star, acq_value = optimize_acqf(lcb_fun, bounds, q = 1, num_restarts = 5, raw_samples = 50)
t2 = time.time()
print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1)





