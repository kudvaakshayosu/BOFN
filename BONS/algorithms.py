#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:44:29 2024
Efficient optimization algorithms
@author: kudva.7
"""

from graph_utils import Graph
from typing import Callable
from typing import Optional
import torch
from torch import Tensor
from gp_network_utils import GaussianProcessNetwork
import time
from botorch.optim import optimize_acqf
import copy
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, UpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.acquisition import MCAcquisitionFunction
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.optim.initializers import gen_batch_initial_conditions
import sys
from TSacquisition_functions import ThompsonSampleFunctionNetwork

torch.set_default_dtype(torch.double)


class PosteriorMean(MCAcquisitionFunction):
    """
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        obj = obj.mean(dim=0)[..., 0]
        return obj


##############################################################################
#################### Algorithms follow BOFN Framework ########################
##############################################################################

def BOFN(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2),
           q: int = 1,  # Default batch number is 1
           acq_type = 'qEI', # 'qUCB', 'qEI', or 'qlogEI'
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
        Function network objective function - the "expensive" simulator
    T : float
        Budget of function Network
    beta : Tensor, optional
        The confidence bound parameter. The default is torch.tensor(2)  
        This is only valid for qUCB

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to optimize the acquisition function
            4) Ninit: Number of initial values
            5) T: Evaluation budget

    """
    # Extract meta data from graph
    input_dim = g.nx 
    Ninit = x_init.size()[0]  
    network_to_objective_transform = g.objective_function
    
    num_restarts=10*input_dim
    raw_samples=100*input_dim
    
    
    
    # Instantiate requirements
    bounds =torch.tensor([[0]*(input_dim),[1]*(input_dim)])
    bounds = bounds.type(torch.float)    
    
    time_opt = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(q,input_dim)
    
    # Start the BO Loop
    for t in range(T):   
        t1 = time.time()
        print('Iteration number', t)        
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        
        
        # Acquisition function
        if acq_type == 'qEI' or acq_type == 'qlogEI':
            if acq_type == 'qEI':
                acquisition_function = qExpectedImprovement(
                                        model=model,
                                        best_f= network_to_objective_transform(Y).max().item(),
                                        sampler=qmc_sampler,
                                        objective=network_to_objective_transform)
            elif acq_type == 'qlogEI':
                acquisition_function = qLogExpectedImprovement(
                                        model=model,
                                        best_f= network_to_objective_transform(Y).max().item(),
                                        sampler=qmc_sampler,
                                        objective=network_to_objective_transform)
            else:
                print('Something went wrong!')
                sys.exit()
            
            posterior_mean_function = PosteriorMean(
                                model=model,
                                sampler=qmc_sampler,
                                objective=network_to_objective_transform,
                                )
            
            batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=acquisition_function,
                    bounds=torch.tensor([[0. for i in range(input_dim)], [1. for i in range(input_dim)]]), 
                    q=q,
                    num_restarts= num_restarts,
                    raw_samples=raw_samples,
                )

            x_star, _ = optimize_acqf(
                acq_function=posterior_mean_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5},
            )

            batch_initial_conditions = torch.cat([batch_initial_conditions, x_star.unsqueeze(0)], 0)
            num_restarts += 1 
            
            
            x_star, acq_value = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_initial_conditions=batch_initial_conditions,
            options={"batch_limit": 2})
        
        elif acq_type == 'qUCB':
            acquisition_function = qUpperConfidenceBound(
                                    model=model,
                                    beta = beta,
                                    sampler=qmc_sampler,
                                    objective = network_to_objective_transform)
            
            x_star, acq_value = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 2})
            
        else:
            print('Enter a valid acquisition function for Network')


        # Computation time
        t2 = time.time()
        time_opt.append(t2 - t1)
        
        X_new = x_star
        
        print('Next point to sample', X_new)
        
        if input_dim == 1:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new.squeeze(0)])
    
    output_dict = {'X': X, 'Y': Y, 'Time': time_opt, 'Ninit': Ninit, 'T': T}
    
    return output_dict


##############################################################################
##########################            BONS   #################################
##############################################################################

def BONS(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           q: int = 1,  # Default batch number is 1
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
        Function network objective function - the "expensive" simulator
    T : float
        Budget of function Network
    beta : Tensor, optional
        The confidence bound parameter. The default is torch.tensor(2)  
        This is only valid for qUCB

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to optimize the acquisition function
            4) Ninit: Number of initial values
            5) T: Evaluation budget

    """
    # Extract meta data from graph
    input_dim = g.nx 
    Ninit = x_init.size()[0]  
    network_to_objective_transform = g.objective_function
    
    num_restarts=10*input_dim
    raw_samples=100*input_dim
    
    
    
    # Instantiate requirements
    bounds =torch.tensor([[0]*(input_dim),[1]*(input_dim)])
    bounds = bounds.type(torch.float)    
    
    time_opt = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    # Start the BO Loop
    # Extract meta data from graph
    input_dim = g.nx 
    Ninit = x_init.size()[0]  
    
    num_restarts=10*input_dim
    raw_samples=100*input_dim
    
    
    
    # Instantiate requirements
    bounds =torch.tensor([[0]*(input_dim),[1]*(input_dim)])
    bounds = bounds.type(torch.float)    
    
    time_opt = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(q,input_dim)
    
    # Start the BO Loop
    for t in range(T):   
        t1 = time.time()
        print('Iteration number', t)        
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        
        acquisition_function = ThompsonSampleFunctionNetwork(model)
        # Sampler
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler)
        
        # batch_initial_conditions = gen_batch_initial_conditions(
        #         acq_function=acquisition_function,
        #         bounds=torch.tensor([[0. for i in range(g.nx)], [1. for i in range(g.nx)]]), 
        #         q=q,
        #         num_restarts= 100,
        #         raw_samples=1000,
        #     )


        x_star, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds= torch.tensor([[0. for i in range(g.nx)], [1. for i in range(g.nx)]]),
            q=q ,
            num_restarts=2,
            raw_samples=5,
            #batch_initial_conditions= batch_initial_conditions,
            #options={"batch_limit": 5},
        )

        # Computation time
        t2 = time.time()
        time_opt.append(t2 - t1)
        
        X_new = x_star
        
        print('Next point to sample', X_new)
        
        if input_dim == 1:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new.squeeze(0)])
    
    output_dict = {'X': X, 'Y': Y, 'Time': time_opt, 'Ninit': Ninit, 'T': T}
    
    return output_dict



















