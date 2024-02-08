#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:58:50 2024
This python file contains all the algorithms required for the 
@author: kudva.7
"""

from graph_utils import Graph
from typing import Callable
import torch
from torch import Tensor
from gp_network_utils import GaussianProcessNetwork
import time
from botorch.optim import optimize_acqf
from acquisition_functions import BONSAI_Acquisition, ARBO_UCB
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
import copy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model 
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)


def BONSAI(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2)
           ) -> dict:
    """
    Parameters
    ----------
    x_init : Tensor
        Initial values of features
    y_init : Tensor
        Initial values of mappings
    g : Graph
        Function network graph structure
    objective : Callable
        Function network objective function
    T : float
        Budget of function Network
    beta : Tensor, optional
        The confidence bound parameter. The default is torch.tensor(2).

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to compute z_star = max_{z} min_{w} max_{eta} UCB(z,w,eta)
            4) Time to compute w_star = min_{w} min_{eta} LCB(z_star,w,eta)

    """
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes

    
    # Instantiate requirements
    bounds_z =torch.tensor([[0]*(nz),[1]*(nz)])
    bounds_z = bounds_z.type(torch.float)
    
    bounds_w =torch.tensor([[0]*(nw),[1]*(nw)])
    bounds_w = bounds_w.type(torch.float)
    
    time_opt1 = []
    time_opt2 = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for t in range(T):   
        
        print('Iteration number', t)        
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        
        # Alternating bound acquisitions
        # 1) max_{z} min_{w} max_{eta} UCB(z,w,eta)
        ucb_fun = BONSAI_Acquisition(model, beta = beta)        
        t1 = time.time()
        z_star, acq_value = optimize_acqf(ucb_fun, bounds_z , q = 1, num_restarts = 10, raw_samples = 50)
        t2 = time.time()
        #print("Time taken for max_{z} min_{w} max_{eta} UCB = ", t2 - t1)
        time_opt1.append(t2 - t1)
        
        #2) min_{w} min_{eta} LCB(z_star,w,eta)
        lcb_fun = BONSAI_Acquisition(model, beta = beta, maximize = False, fixed_variable = z_star)       
        t1 = time.time()
        w_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 10, raw_samples = 50)
        t2 = time.time()
        #print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1) 
        time_opt2.append(t2 - t1)
        
        # Store the new point to sample
        X_new[..., design_input] = z_star
        X_new[..., uncertainty_input] = w_star
        
        print('Next point to sample', X_new)
        
        Y_new = objective(X_new.unsqueeze(0))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])
    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'T2': time_opt2}
    
    return output_dict
    


def ARBO(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2)) -> dict:
    """
    Parameters
    ----------
    x_init : Tensor
        Initial values of features
    y_init : Tensor
        Initial values of mappings
    g : Graph
        Function network graph structure
    objective : Callable
        Function network objective function
    T : float
        Budget of function Network
    beta : Tensor, optional
        The confidence bound parameter. The default is torch.tensor(2).

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to compute z_star = max_{z} min_{w} max_{eta} UCB(z,w,eta)
            4) Time to compute w_star = min_{w} min_{eta} LCB(z_star,w,eta)

    """
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    input_index_info = [design_input,uncertainty_input] # Needed for ARBO acquisition function

    
    # Instantiate requirements
    bounds_z =torch.tensor([[0]*(nz),[1]*(nz)])
    bounds_z = bounds_z.type(torch.float)
    
    bounds_w =torch.tensor([[0]*(nw),[1]*(nw)])
    bounds_w = bounds_w.type(torch.float)
    
    time_opt1 = []
    time_opt2 = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for t in range(T):     
        print('Iteration number', t) 
        #covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=input_dim)) 
        model = SingleTaskGP(X, Y[...,-1].unsqueeze(-1), outcome_transform=Standardize(m=1))
        
        
        mlls = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mlls)
        
        model.eval()
  
        # Alternating bound acquisitions
        # 1) max_{z} min_{w} UCB(z,w)
        
        ucb_fun = ARBO_UCB( model, beta = beta, input_indices = input_index_info)
        
        t1 = time.time()
        z_star, acq_value = optimize_acqf(ucb_fun, bounds_z, q = 1, num_restarts = 10, raw_samples = 1000)
        t2 = time.time()
        #print("Time taken for max_{z} min_{w} UCB(z,w) = ", t2 - t1)
        time_opt1.append(t2 - t1)
        
        lcb_fun = ARBO_UCB( model, beta = beta, input_indices = input_index_info, maximize = False, fixed_variable= z_star)
        t1 = time.time()
        w_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 10, raw_samples = 1000)
        t2 = time.time()
        #print("Time taken for  min_{w} LCB(w; z_star) = ", t2 - t1)
        time_opt2.append(t2 - t1)        
        
        # Store the new point to sample
        X_new[..., design_input] = z_star
        X_new[..., uncertainty_input] = w_star
        
        print('Next point to sample', X_new)
        
        Y_new = objective(X_new.unsqueeze(0))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])      
        
        ##############################################################
        # # Test to see if optimizers are looking okay - Check
        
        # test_z = torch.arange(0, 1, (1)/100)
        # test_w = torch.arange(0, 1, (1)/100)
        # # create a mesh from the axis
        # x2, y2 = torch.meshgrid(test_z, test_w)

        # # reshape x and y to match the input shape of fun
        # xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)
        
        
        # posterior = model.posterior(xy)
        # ucb = posterior.mean + posterior.variance.sqrt()
        
        # ucb = ucb.reshape(x2.size())
        
        # fig, ax = plt.subplots(1, 1)
        # plt.set_cmap("jet")
        # contour_plot = ax.contourf(x2,y2,ucb.detach().numpy())
        # fig.colorbar(contour_plot)
        # plt.xlabel('z')
        # plt.ylabel('w')
        
        ########################################################################    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'T2': time_opt2}
    
    return output_dict
    
    