#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:38:41 2024

@author: kudva.7
"""

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction

from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.normal import IIDNormalSampler
# import botorch.sampling.samplers import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from typing import Optional
import torch
import sys
from botorch.utils.safe_math import smooth_amax, smooth_amin

from gp_network_utils import MultivariateNormalNetwork


class BONSAI_Acquisition(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        maximize: bool = True,
        fixed_variable = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.n_nodes = self.model.dag.n_nodes
        self.active_input_indices = self.model.dag.active_input_indices
        self.uncertain_indices = self.model.dag.uncertain_input_indices
        self.design_indices = self.model.dag.design_input_indices
        self.fixed_variable = fixed_variable

        # self.n = 1

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
        # means, sigma = posterior.mean_sigma
        if self.maximize:
            assert X.size()[-1] == self.model.dag.nz, f"input dimension must be {self.model.dag.nz}, current input dimension X.size()[-1]"
            val = BONSAI_Confidence_Alternator(X, model=self.model, beta=self.beta, maximize=self.maximize)
            val = val.forward_ucb()
        else:
            if self.fixed_variable is None:
                print('Fixed variable needs to be specified if maximize == "False"')                
            assert X.size()[-1] == self.model.dag.nw, f"input dimension must be {self.model.dag.nz}, current input dimension X.size()[-1]"
            val = BONSAI_Confidence_Alternator(X, model=self.model, beta=self.beta, maximize=self.maximize, fixed_variable = self.fixed_variable)
            val = val.forward_lcb()

        return val


class BONSAI_Confidence_Alternator():
    def __init__(self,
                 F: Tensor, # The free variable
                 model: Model,
                 beta: Tensor,
                 maximize: bool = True,
                 fixed_variable = None
                 ) -> None:
        """
        This is a wrapper that uses alternating bounds
        """

        # Obtain the X values
        self.model = model
        
        self.nx = self.model.dag.nx
        self.nw = self.model.dag.nw
        self.nz = self.model.dag.nz
        self.n_nodes = self.model.dag.n_nodes
        self.beta = beta
        self.Neta = 500
        self.fixed_variable = fixed_variable
        
        if maximize: # for max min max ucb
            self.Z = F
            self.Nz = F.size()[0]
            self.Nw = 500
            
            
        else: # for max max -lcb
            self.W = F  
            self.Nw = F.size()[0]

    def forward_ucb(self):
        # Create Xe corresponds to z + w + eta
        X = torch.empty(self.Nz*self.Nw*self.Neta, self.nz + self.nw)

        torch.manual_seed(10000)
        # Insert the testing points:
        X[..., self.model.dag.design_input_indices] = self.Z.squeeze(-2).repeat_interleave(self.Nw*self.Neta, dim=0)
        X[..., self.model.dag.uncertain_input_indices] = torch.rand(self.Nw, self.nw).repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)
        
        # Create the inter-model calibertaion term
        eta = (torch.rand(self.Neta, self.n_nodes) * 2 - 1).repeat(self.Nw*self.Nz, 1)
        
        posterior = self.model.posterior(X)
        ucb_vals = posterior.Bonsai_UCB(
            eta=eta, maximize=True, beta=self.beta)
        # Assume that the final node is the output of the torch tensor
        ucb_vals = ucb_vals[..., -1]

        # Create an empty tensor of meshgrids of size Nz X Nw X Neta
        ucb_mesh = torch.empty(self.Nz, self.Nw, self.Neta)

        for i in range(self.Nz):
            ucb_mesh[i] = ucb_vals[i*self.Nw*self.Neta:(i + 1)*self.Nw*self.Neta].reshape(self.Nw,self.Neta)
            
        objective = smooth_amin(smooth_amax(ucb_mesh, dim = -1), dim = -1)

        return objective

    def forward_lcb(self):
        X = torch.empty(self.Nw*self.Neta, self.nz + self.nw)
        # Insert the testing points:
        X[..., self.model.dag.design_input_indices] = self.fixed_variable.repeat_interleave(self.Nw*self.Neta, dim=0)
        X[..., self.model.dag.uncertain_input_indices] = self.W.squeeze(-2).repeat_interleave(self.Neta,dim = 0)
        
        torch.manual_seed(10000)
        eta = (torch.rand(self.Neta, self.n_nodes) * 2 - 1).repeat(self.Nw, 1)
        
        posterior = self.model.posterior(X)
        lcb_vals = posterior.Bonsai_UCB(
            eta=eta, maximize=False, beta=self.beta)
        
        # Assume that the final node is the output of the torch tensor
        lcb_vals = lcb_vals[..., -1]
        
        # Create a mesh of size Nw X Neta
        lcb_mesh = torch.empty(self.Nw, self.Neta)
        
        for i in range(self.Nw):
            lcb_mesh[i] = lcb_vals[i*self.Neta:(i + 1)*self.Neta].reshape(self.Neta)
        
        objective = smooth_amax(lcb_mesh, dim = -1)
        
        return objective
        
        