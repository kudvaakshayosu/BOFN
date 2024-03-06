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
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
import copy
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, UpperConfidenceBound, LogExpectedImprovement
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model 
import matplotlib.pyplot as plt
from utils import round_to_nearest_set
import copy
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.acquisition import MCAcquisitionFunction, AnalyticAcquisitionFunction
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.optim.initializers import gen_batch_initial_conditions
import sys

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
    
class BONS_Acquisition(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.n_nodes = self.model.dag.n_nodes
        self.nx = self.model.dag.nx

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X using scalarization

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        # Remove unnecessary dimensions just like analytical acq function
        # self.n += 1
        # print(self.n)
        self.beta = self.beta.to(X)
        
        # Seperate the objective into two
        x = X[...,0:self.nx]
        eta = X[...,self.nx:]*2 - 1
        
        # Obtain posterior
        posterior = self.model.posterior(x)
        ucb_vals = posterior.Bonsai_UCB(eta=eta, maximize=True, beta=self.beta)
        
        objective = self.model.dag.objective_function(ucb_vals)
        objective = objective.squeeze(-2).squeeze(-1)
        return objective


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
           acq_type = 'qEI', 
           nominal_mode = False
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
                                    objective=network_to_objective_transform)
            
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
        
        if nominal_mode:
            return X_new
        else:       
            # Append the new values
            X = torch.vstack([X,X_new])
            print('Next point to sample', X_new)
            
            if input_dim == 1:
                Y_new = objective(X_new.unsqueeze(0))
            else:
                Y_new = objective(copy.deepcopy(X_new))
        Y = torch.vstack([Y,Y_new.squeeze(0)])
    
    output_dict = {'X': X, 'Y': Y, 'Time': time_opt, 'Ninit': Ninit, 'T': T}
    
    return output_dict

###############################################################################
##################  Black-box Bayesian Optimization ###########################
###############################################################################

def BayesOpt(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2),
           q: int = 1,  # Default batch number is 1
           acq_type = 'qEI'
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
            3) Time to optimize the acquisition function
            4) Ninit: Number of initial values
            5) T: Evaluation budget

    """
    # Extract meta data from graph
    active_input_indices = g.active_input_indices  
    input_dim = g.nx 
    n_nodes = g.n_nodes
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
    for t in range(T):   
        
        print('Iteration number', t)   
        model = SingleTaskGP(X, network_to_objective_transform(Y).unsqueeze(-1),train_Yvar = torch.ones(network_to_objective_transform(Y).unsqueeze(-1).shape)*1e-4, outcome_transform=Standardize(m=1))       
        mlls = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mlls)
        
        if g.custom_hyperparameters:           
            model.covar_module.outputscale = g.output_scale
            model.covar_module.base_kernel.lengthscale = g.length_scale
        
        model.eval()
        
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        
        t1 = time.time()
        if acq_type == 'qEI':
            acquisition_function = qExpectedImprovement(
                                    model=model,
                                    best_f= network_to_objective_transform(Y).max().item(),
                                    sampler=qmc_sampler)
        elif acq_type == 'qlogEI':
            acquisition_function = qLogExpectedImprovement(
                                    model=model,
                                    best_f= network_to_objective_transform(Y).max().item(),
                                    sampler=qmc_sampler)
        elif acq_type == 'qUCB':
            acquisition_function = qUpperConfidenceBound(
                                    model=model,
                                    beta = beta,
                                    sampler=qmc_sampler)
            
        elif acq_type == 'UCB':
            acquisition_function = qUpperConfidenceBound(
                                    model=model,
                                    beta = beta)
        
        elif acq_type == 'logEI':
            acquisition_function = LogExpectedImprovement(model = model, best_f = network_to_objective_transform(Y).max().item())
        
        x_star, acq_value = optimize_acqf(acq_function=acquisition_function,
                                        bounds=bounds,
                                        q=q,
                                        num_restarts=num_restarts,
                                        raw_samples=raw_samples,
                                        options={"batch_limit": 2})
        
        # Computation time
        t2 = time.time()
        time_opt.append(t2 - t1)
        
        X_new = x_star
        
        print('Next point to sample', X_new)
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        try:
            Y = torch.vstack([Y,Y_new.squeeze(0)])
        except:
            Y = torch.vstack([Y,Y_new.squeeze(-2).squeeze(-1)])
    
    output_dict = {'X': X, 'Y': Y, 'Time': time_opt, 'Ninit': Ninit, 'T': T}
    
    return output_dict

###############################################################################
############# BONS : Bayesian Optimization of Network systems #################
###############################################################################

def BONS(x_init: Tensor,
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
            3) Time to optimize the acquisition function
            4) Ninit: Number of initial values
            5) T: Evaluation budget

    """
    # Extract meta data from graph
    input_dim = g.nx 
    Ninit = x_init.size()[0]  
    network_to_objective_transform = g.objective_function
    n_nodes = g.n_nodes
    
    num_restarts=10*input_dim
    raw_samples=100*input_dim
    
    
    
    # Instantiate requirements
    bounds =torch.tensor([[0]*(input_dim + n_nodes),[1]*(input_dim + n_nodes)])
    bounds = bounds.type(torch.float)    
    
    time_opt = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    # Start the BO Loop
    for t in range(T):   
        t1 = time.time()
        print('Iteration number', t)        
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        
        acquisition_function = BONS_Acquisition(model, beta = beta)  
        
        # Optimize the acquisition function
        t1 = time.time()
        x_star, acq_value = optimize_acqf(acquisition_function, bounds , q = 1, num_restarts = num_restarts, raw_samples = raw_samples)


        # Computation time
        t2 = time.time()
        time_opt.append(t2 - t1)
        
        X_new = x_star[...,0:input_dim]
        
        print('Next point to sample', X_new)
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        try:
            Y = torch.vstack([Y,Y_new.squeeze(0)])
        except:
            Y = torch.vstack([Y,Y_new.squeeze(-2).squeeze(-1)])
    output_dict = {'X': X, 'Y': Y, 'Time': time_opt, 'Ninit': Ninit, 'T': T}
    
    return output_dict
















    