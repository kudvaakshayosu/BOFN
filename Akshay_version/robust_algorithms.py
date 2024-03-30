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
from acquisition_functions import BONSAI_Acquisition, ARBO_UCB, ARBONS_Acquisition, BONSAINAB_Acquisition
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
import copy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model 
import matplotlib.pyplot as plt
from utils import round_to_nearest_set
from single_level_algorithms import BOFN
import sys
from TSacquisition_functions import ThompsonSampleFunctionNetwork, GPNetworkThompsonSampler, maxmin_ThompsonSampleFunctionNetwork
from fixed_feature import FixedFeatureAcquisitionNetworkFunction
from botorch.models.model import Model


def remove_numbers_from_list_of_lists(numbers_to_remove, list_of_lists):
    return [list(filter(lambda x: x not in numbers_to_remove, sublist)) for sublist in list_of_lists]




def min_W_TS(model: Model,
             z_star: Tensor):
    
    X_new = torch.empty(model.dag.w_num_combinations, model.dag.nx)
    
    X_new[...,model.dag.design_input_indices] = z_star.repeat(model.dag.w_num_combinations, 1)
    X_new[...,model.dag.uncertain_input_indices] = model.dag.w_combinations  
    
    
    ts_network = GPNetworkThompsonSampler(model)
    ts_network.create_sample()
    
    Y_new = ts_network.query_sample(X_new)
    
    Y_new = model.dag.objective_function(Y_new)
   
    # Outer-max
    Y_maxmin = Y_new.min(0).values
    maxmin_idx = Y_new.min(0).indices
    
    w_final = model.dag.w_combinations[maxmin_idx]
    
    
    return w_final.unsqueeze(0), Y_maxmin
    


###############################################################################
### BONSAI - Bayesian Optimization of Network Systems under uncertAInty #######
###############################################################################

def BONSAI(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
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
        The confidence bound parameter. The default is torch.tensor(2)  

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to compute z_star = max_{z} min_{w} max_{eta} UCB(z,w,eta)
            4) Time to compute w_star = min_{w} min_{eta} LCB(z_star,w,eta)
            5) Ninit: Number of initial values
            6) T: Evaluation budget

    """
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    Ninit = x_init.size()[0]    
    
    
    # Instantiate requirements
    bounds =torch.tensor([[0]*(nz + nw),[1]*(nz + nw)])
    bounds = bounds.type(torch.float)
    

    time_opt1 = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for t in range(T):   
        
        print('Iteration number', t)        
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        
        if nw == 0:
            ts_fun = ThompsonSampleFunctionNetwork(model)
            
            t1 = time.time()
           
            z_star, acq_val = optimize_acqf(ts_fun, bounds, q = 1, num_restarts = 100, raw_samples = 1000)      
                  
            
           # Step 2) min TS(z_star, w))       
            w_star = 0   
            
            t2 = time.time()
            
            time_opt1.append(t2 - t1)
            
            X_new = z_star           
            
            
        else:
        
            # 1)  Step 1) max min TS(z, w)
            ts_fun = maxmin_ThompsonSampleFunctionNetwork(model)
            t1 = time.time()
           
            z_star, acq_val = optimize_acqf(ts_fun, bounds, q = 1, num_restarts = 100, raw_samples = 1000)      
            
            # Get the values that are important
            z_star = z_star[..., g.design_input_indices]
                  
            
           # Step 2) min TS(z_star, w))       
            w_star, acq_val = min_W_TS(model = model, z_star = z_star)       
            
            t2 = time.time()
            #print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1) 
            
            time_opt1.append(t2 - t1)
            
            # Store the new point to sample
            X_new[..., design_input] = z_star
            X_new[..., uncertainty_input] = w_star
            
           
        print('Next point to sample', X_new) 
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        print('Objective function here is')
        print(Y_new)
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])
    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'Ninit': Ninit, 'T': T}
    
    return output_dict



def BONSAI_Recommendor(data, # This is a pickle folder 
           g: Graph,
           beta = torch.tensor(2)
           ) -> dict:

    """
    Recommends data every iteration
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
    bounds_w = bounds_w.type(torch.float)   
    
    T = data['T']
    Ninit = data['Ninit']
    X = data['X'][:Ninit]
    Y = data['Y'][:Ninit]
    
    Y_out = torch.empty(T)
    Z_out = torch.empty(T, g.nz)
    
    
    for i in range(T):
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)       
        Z_test = data['X'][Ninit: Ninit + i + 1,g.design_input_indices]
        Y_test = torch.zeros(Z_test.size()[0])
        print('Iteration', i)
        
        for j in range(Z_test.size()[0]): 
            z_star = Z_test[j]
            lcb_fun = BONSAI_Acquisition(model, beta = beta, maximize = False, fixed_variable = z_star.unsqueeze(0))
            w_eta_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 100, raw_samples = 1000)
            Y_test[j] = -1*acq_value
        
        Y_out[i] = Y_test[:Ninit + i + 1].max()
        Z_out[i] = Z_test[Y_test[:Ninit + i + 1].argmax()]
        print('BONSAI Recommendor suggested best point as', Z_out[i])
        print('Robust LCB at this recommendation', Y_out[i])
        
        X = data['X'][:Ninit + i + 1]
        Y = data['Y'][:Ninit + i + 1]
        
    
    # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict

def Mean_Recommendor_final(data, # This is a pickle folder 
           g: Graph,
           beta = torch.tensor(2),
           T= None
           ) -> dict:

    """
    Recommends design variables at the end of each iteration
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
    bounds_w = bounds_w.type(torch.float)   
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)
    
   
    
    
    
    if g.nw == 0:
        
        Z_test = data['X'][Ninit: Ninit + T + 1, :nz]
        Y_test = data['Y'][Ninit: Ninit + T + 1]
        Y_out_val = g.objective_function(Y_test).values
        Z_out = Z_test[Y_out_val.argmax()]     
        Y_out = Y_out_val.max()
        
    else:
        
        Z_test = data['X'][Ninit: Ninit + T + 1, g.design_input_indices]
        Y_test = torch.zeros(Z_test.size()[0])
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)        
        
        # Mean recommender       
        for j in range(Z_test.size()[0]): 
            z_star = Z_test[j]
            z_star = z_star.repeat(g.w_num_combinations,1)        
            X = torch.hstack((z_star, g.w_combinations))        
            posterior = model.posterior(X)
            mean, _ = posterior.mean_sigma
            Y_test[j] = mean.min().detach()
            
        Y_out = Y_test.max()
        Z_out = Z_test[Y_test[:Ninit + T + 1].argmax()]
        
        print('BONSAI Recommendor suggested best point as', Z_out)
        print('Robust LCB at this recommendation', Y_out)
        
        # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict



# def BONSAI_Recommendor_final(data, # This is a pickle folder 
#            g: Graph,
#            beta = torch.tensor(2),
#            T= None
#            ) -> dict:

#     """
#     Recommends design variables at the end of each iteration
#     """
#     uncertainty_input = g.uncertain_input_indices
#     design_input = g.design_input_indices   
#     nz = g.nz
#     nw = g.nw
#     input_dim = g.nx 
#     n_nodes = g.n_nodes
    
#     # Instantiate requirements
    
#     bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
#     bounds_w = bounds_w.type(torch.float)   
    
    
#     Ninit = data['Ninit']
#     X = data['X']
#     Y = data['Y']
    
#     if T:
#         print('Pre-defined T value')
#     else:
#         T = Y[Ninit:].size()[0]
    
#     Y_out = torch.empty(1,1)
#     Z_out = torch.empty(1, g.nz)
    
#     model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
#     Z_test = data['X'][Ninit: Ninit + T + 1, g.design_input_indices]
#     Y_test = torch.zeros(Z_test.size()[0])

#     #         
#     for j in range(Z_test.size()[0]): 
#         z_star = Z_test[j]
#         lcb_fun = BONSAI_Acquisition(model, beta = beta, maximize = False, fixed_variable = z_star.unsqueeze(0))
#         w_eta_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 100, raw_samples = 2000)
#         Y_test[j] = -1*acq_value
        
#     Y_out = Y_test.max()
#     Z_out = Z_test[Y_test[:Ninit + T + 1].argmax()]
    
#     print('BONSAI Recommendor suggested best point as', Z_out)
#     print('Robust LCB at this recommendation', Y_out)
    
#     # Acquisition function values
#     output_dict = {'Z':Z_out , 'Y': Y_out}
    
#     return output_dict

################################################################################
###################   Nominal mode   log EI Network ##############################
##################################################################################
def BOFN_nominal_mode(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2),
           acq_fun = 'qlogEI',
           nominal_w = None,
           ) -> dict:
    
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    Ninit = x_init.size()[0]
    
    time_opt1 = []
    
    # Manipulate the graph to a nominal version of the problem
    
    if not nw == 0:
        g_new = copy.deepcopy(g)
        new_active_input_indices = remove_numbers_from_list_of_lists(g_new.uncertain_input_indices, g_new.active_input_indices)
        g_new.register_active_input_indices(new_active_input_indices)
    else:
        g_new = g 
    
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for i in range(T):  
        print('Iteration number', i)
        
        t1 = time.time()
        z_star = BOFN(X, Y, g_new, objective = None, T = 1, acq_type = acq_fun, nominal_mode = True)
        t2 = time.time()
        time_opt1.append(t2 - t1)
        
        w_star = nominal_w
        
        # Store the new point to sample
        if nw == 0:
            X_new = z_star
        else:
        
            X_new[..., design_input] = z_star
            X_new[..., uncertainty_input] = w_star
        
        print('New point sampled is ', X_new)
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
            
        print('Objective value is ', g.objective_function(Y_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])
    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'Ninit': Ninit, 'T': T}
    
    return output_dict


def BOFN_Recommendor_final(data, # This is a pickle folder 
           g: Graph,
           beta = torch.tensor(2),
           T= None
           ) -> dict:

    """
    Recommends design variables at the end of each iteration
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
    bounds_w = bounds_w.type(torch.float)   
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)
    
    model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
    Z_test = data['X'][Ninit: Ninit + T + 1, g.design_input_indices]
    Y_test = g.objective_function(data['Y'][Ninit: Ninit + T + 1])
        
    Y_out = Y_test.max()
    Z_out = Z_test[Y_test.argmax()]
    
    print('BOFN Recommendor suggested best point as', Z_out)
    print('Robust LCB at this recommendation', Y_out)
    
    # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict
    




  
#################################################################################
##########################  For Ablation studies ###############################
################################################################################


# TODO: Recommendation for ARBO
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
    
    if nw == 0:
        nw = g.w_combinations.size()[1]
        uncertainty_input = [i for i in range(nz,nz + nw)]
        design_input = [i for i in range(nz)]
    
    input_dim = nz + nw
    n_nodes = g.n_nodes
    input_index_info = [design_input,uncertainty_input] # Needed for ARBO acquisition function
    w_combinations = g.w_combinations
    Ninit = x_init.size()[0]

    
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
        try:
            model = SingleTaskGP(X, g.objective_function(Y).unsqueeze(-1), outcome_transform=Standardize(m=1))
        except:
            model = SingleTaskGP(X, g.objective_function(X,Y).unsqueeze(-1), outcome_transform=Standardize(m=1))
        
        
        mlls = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mlls)
        
        model.eval()
  
        # Alternating bound acquisitions
        # 1) max_{z} min_{w} UCB(z,w)
        
        ucb_fun = ARBO_UCB( model, beta = beta, input_indices = input_index_info, w_combinations = w_combinations)
        
        t1 = time.time()
        z_star, acq_value = optimize_acqf(ucb_fun, bounds_z, q = 1, num_restarts = 10, raw_samples = 100)
        t2 = time.time()
        #print("Time taken for max_{z} min_{w} UCB(z,w) = ", t2 - t1)
        time_opt1.append(t2 - t1)
        
        lcb_fun = ARBO_UCB( model, beta = beta, input_indices = input_index_info, maximize = False, fixed_variable= z_star, w_combinations= w_combinations)
        t1 = time.time()
        w_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 10, raw_samples = 100)
        
        # Seperate the discrete values
        if w_combinations is None:
            w_star = w_star
        else:
            w_star = round_to_nearest_set(w_star, g.w_sets)
            
        t2 = time.time()
        #print("Time taken for  min_{w} LCB(w; z_star) = ", t2 - t1)
        time_opt2.append(t2 - t1)        
        
        # Store the new point to sample
        X_new[..., design_input] = z_star
        X_new[..., uncertainty_input] = w_star
        
        print('Next point to sample', X_new)
        
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
            
        print('Value of sample obtained', Y_new)
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])      
        
        #############################################################
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
        
        #######################################################################    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'T2': time_opt2 ,'Ninit': Ninit, 'T': T}
    
    return output_dict


def ARBO_Recommendor_final(data, # This is a pickle folder 
           g: Graph,
           beta = torch.tensor(2),
           T = None,
           ) -> dict:

    """
    Recommends design variables at the end of each iteration
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw),[1]*(nw)])
    bounds_w = bounds_w.type(torch.float)   
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)
    
    model = SingleTaskGP(X, g.objective_function(Y).unsqueeze(-1), outcome_transform=Standardize(m=1))
    Z_test = data['X'][Ninit: Ninit + T + 1, g.design_input_indices]
    Y_test = torch.zeros(Z_test.size()[0])

    #         
    for j in range(Z_test.size()[0]): 
        z_star = Z_test[j].unsqueeze(0)      
        lcb_fun = ARBO_UCB( model, beta = beta, input_indices = [g.design_input_indices, g.uncertain_input_indices], maximize = False, fixed_variable= z_star, w_combinations= g.w_combinations)
        w_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 10, raw_samples = 1000)
        w_star = round_to_nearest_set(w_star, g.w_sets)
        
        Y_test[j] = -1*acq_value
        
    Y_out = Y_test.max()
    Z_out = Z_test[Y_test[:Ninit + T + 1].argmax()]
    
    print('BONSAI Recommendor suggested best point as', Z_out)
    print('Robust LCB at this recommendation', Y_out)
    
    # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict





















######################################################################################
######## Adversarially Robust Bayesian Optimization for Network Systems #############
####################################################################################

def ARBONS(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2),
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
        The confidence bound parameter. The default is torch.tensor(2)  

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to compute z_star = max_{z} min_{w} max_{eta} UCB(z,w,eta)
            4) Time to compute w_star = min_{w} min_{eta} LCB(z_star,w,eta)
            5) Ninit: Number of initial values
            6) T: Evaluation budget

    """
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    Ninit = x_init.size()[0]
    #input_index_info = [design_input,uncertainty_input]
    #w_combinations = g.w_combinations
    Ninit = x_init.size()[0] 
    
    
    
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
        ucb_fun = ARBONS_Acquisition(model, beta = beta)      
        t1 = time.time()
        z_star, acq_value = optimize_acqf(ucb_fun, bounds_z , q = 1, num_restarts = 10, raw_samples = 50)
        t2 = time.time()
        #print("Time taken for max_{z} min_{w} max_{eta} UCB = ", t2 - t1)
        time_opt1.append(t2 - t1)
        
        #2) min_{w} min_{eta} LCB(z_star,w,eta)
        lcb_fun = ARBONS_Acquisition( model, beta = beta, maximize = False, fixed_variable= z_star)     
        t1 = time.time()
        w_eta_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 100, raw_samples = 500)
        
        # Seperate the discrete values
        if g.w_combinations is None:
            w_star = w_eta_star[:,0:nw]
        else:
            w_star = round_to_nearest_set(w_eta_star[:,0:nw], g.w_sets)
        t2 = time.time()
        #print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1) 
        time_opt2.append(t2 - t1)
        
        # Store the new point to sample
        X_new[..., design_input] = z_star
        X_new[..., uncertainty_input] = w_star
        
        print('Next point to sample', X_new)
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])
    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'T2': time_opt2, 'Ninit': Ninit, 'T': T}
    
    return output_dict


#################################################################################
########################### BONSAI No Alternating Bounds ######################
#################################################################################








    
#def BONSAI(x_init: Tensor,
#            y_init: Tensor, 
#            g: Graph,
#            objective: Callable,
#            T: float,
#            beta = torch.tensor(2),
#            ) -> dict:
#     """
#     Parameters
#     ----------
#     x_init : Tensor
#         Initial values of features
#     y_init : Tensor
#         Initial values of mappings
#     g : Graph
#         Function network graph structure
#     objective : Callable
#         Function network objective function
#     T : float
#         Budget of function Network
#     beta : Tensor, optional
#         The confidence bound parameter. The default is torch.tensor(2)  

#     Returns
#     -------
#     output_dict: dict
#         Constains the following data:
#             1) All X values
#             2) All Y values
#             3) Time to compute z_star = max_{z} min_{w} max_{eta} UCB(z,w,eta)
#             4) Time to compute w_star = min_{w} min_{eta} LCB(z_star,w,eta)
#             5) Ninit: Number of initial values
#             6) T: Evaluation budget

#     """
#     # Extract meta data from graph
#     uncertainty_input = g.uncertain_input_indices
#     design_input = g.design_input_indices   
#     nz = g.nz
#     nw = g.nw
#     input_dim = g.nx 
#     n_nodes = g.n_nodes
#     Ninit = x_init.size()[0] 
    
#     # Instantiate requirements
#     bounds_z =torch.tensor([[0]*(nz),[1]*(nz)])
#     bounds_z = bounds_z.type(torch.float)
    
#     bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
#     bounds_w = bounds_w.type(torch.float)
    
#     time_opt1 = []
#     time_opt2 = []
    
#     # Create a vector for collecting training data
#     X = copy.deepcopy(x_init)
#     Y = copy.deepcopy(y_init)
    
#     X_new = torch.empty(input_dim)
#     X_old = torch.empty(input_dim,1)
#     repeat_count = 0
    
    
#     for t in range(T):   
        
#         print('Iteration number', t)        
#         model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        
#         """
#         # Alternating bound acquisitions
#         # 1) max_{z} min_{w} max_{eta} UCB(z,w,eta)
#         ucb_fun = BONSAI_Acquisition(model, beta = beta)        
#         t1 = time.time()
#         z_star, acq_value = optimize_acqf(ucb_fun, bounds_z , q = 1, num_restarts = 10, raw_samples = 50)
#         t2 = time.time()
#         #print("Time taken for max_{z} min_{w} max_{eta} UCB = ", t2 - t1)
#         time_opt1.append(t2 - t1)        
#         """
        
#         #1) max_{z} min_{w} Graph Thompson
#         t1 = time.time()
#         z_star, acq_val_thomp, _ =  ThompsonSampleFunctionNetworkRobust(model = model)
#         t2 = time.time()
#         time_opt1.append(t2 - t1)        
        
        
#         #2) min_{w} min_{eta} LCB(z_star,w,eta)
#         lcb_fun = BONSAI_Acquisition(model, beta = beta, maximize = False, fixed_variable = z_star)       
#         t1 = time.time()
#         w_eta_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 5, raw_samples = 500)
        
#         print('LCB value is', -1*acq_value)
        
#         print('w_eat_star',w_eta_star[:,nw:] )
#         # Seperate the discrete values
#         if g.w_combinations is None:
#             w_star = w_eta_star[:,0:nw]
#         else:
#             w_star = round_to_nearest_set(w_eta_star[:,0:nw], g.w_sets)
#         t2 = time.time()
#         #print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1) 
#         time_opt2.append(t2 - t1)
        
#         # Store the new point to sample
#         X_new[..., design_input] = z_star
#         X_new[..., uncertainty_input] = w_star
        
#         print('Next point to sample', X_new)
        
#         try:
#             if input_dim == 2:
#                 Y_new = objective(X_new.unsqueeze(0))
#             else:
#                 Y_new = objective(copy.deepcopy(X_new))
#         except:
#             break
        
#         # Append the new values
#         X = torch.vstack([X,X_new])      
#         Y = torch.vstack([Y,Y_new])
        
        
#         # There are repetaing values so does not make 
#         if torch.norm(X_new - X_old).detach() < 0.01:
#             repeat_count += 1
#         else:
#             repeat_count = 0
        
#         if repeat_count == 3:
#             break       
        
    
#     output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'T2': time_opt2, 'Ninit': Ninit, 'T': T}
    
#     return output_dict