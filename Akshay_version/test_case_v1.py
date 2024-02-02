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

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from acquisition_functions import NetworkUCBOptimizer

torch.set_default_dtype(torch.double)

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
 
torch.manual_seed(1000)
x_test =  torch.rand(1,input_dim)  

# Test if everything is okay
y_true = function_network(x_test)   

network_to_objective_transform = lambda Y: Y.sum(dim = -1)
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)

g = Graph(3)
g.addEdge(0, 1)
g.addEdge(1, 2)
active_input_indices = [[0],[1],[0,1]]
g.figure()

# Start the modeling procedure
Ninit = 30
n_outs = g.n_nodes




x_init = torch.rand(Ninit, input_dim)
y_init = torch.zeros(Ninit,n_outs)

for i in range(Ninit):
    y_init[i] = function_network(x_init[i])


model = GaussianProcessNetwork(train_X=x_init, train_Y=y_init, dag=g,
                  active_input_indices=active_input_indices)


test_val = torch.rand(2,2)

post = model.posterior(test_val)

mean_test, sigma_test = post.mean_sigma

####
#ucb_fun = NetworkUCBOptimizer(model, torch.tensor(2))



#bounds =torch.tensor([[0,0],[1,1]])
#bounds = bounds.type(torch.float)

#new_point, acq_value = optimize_acqf( ucb_fun, bounds, q = 1, num_restarts = 100, raw_samples = 1000)


# qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))

# posterior_mean_function = PosteriorMean(
#     model=model,
#     sampler=qmc_sampler,
#     objective=network_to_objective_transform,
# )


# 

# test_sample = model.posterior(test_val)

#bb = test_sample.rsample()





#mean = PosteriorMean(model = model, maximize = True)

# Start 



#g.figure()   # Plot figure to make sure network looks okay



