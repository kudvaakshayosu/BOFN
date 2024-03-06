#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:38:41 2024

@author: kudva.7
"""

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from torch.quasirandom import SobolEngine
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
import copy
from botorch.utils.safe_math import smooth_amax, smooth_amin

from gp_network_utils import MultivariateNormalNetwork

torch.set_default_dtype(torch.double)

############### Helper functions ##############################

def active_corners(theta_min,theta_max):
    """
    This code is mainly used to generate all corners of box constraints.
    Will be incorporated in the Object that will give us the bounds.

    inputs:
    theta_min -- N dimensional tensor
    theta_max -- N dimensional tensor

    output -- 2^(N) X N dimensional tensor
    """
    size_t1 = torch.Tensor.size(theta_min)
    size_t2 = torch.Tensor.size(theta_max)

    # Show error if dimensions dont match:
    if size_t1 != size_t2:
        sys.exit('The dimensions of bounds dont match: Please enter valid inputs')

    val = size_t1[0]
    size_out = 2**(val)
    output = torch.zeros(size_out,val)
    output_iter = torch.zeros(size_out)

    for i in range(val):
        div_size = int(size_out/(2**(i+1)))
        divs = int(size_out/div_size)
        div_count = 0
        for j in range(divs):
            if bool(j%2):
                output_iter[div_count:div_count+div_size] = theta_min[i]*torch.ones(div_size)
            else:
                output_iter[div_count:div_count+div_size] = theta_max[i]*torch.ones(div_size)
            div_count = div_count + div_size
        output[:,i] = output_iter
    return output

def get_nodes_coeffs(n_nodes = None): # 

    """
        Generates extreme eta values for  high dimensional cases
    """
    
    bound1 = torch.ones(n_nodes)
    bound2 = -1*torch.ones(n_nodes)
    
    return active_corners( bound1, bound2)
#############################################################################


class BONSAI_Acquisition(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        maximize: bool = True,
        fixed_variable = None,
        w_vals = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.n_nodes = self.model.dag.n_nodes
        #self.active_input_indices = self.model.dag.active_input_indices
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
                sys.exit()                
            assert X.size()[-1] == self.model.dag.nw + self.model.dag.n_nodes, f"input dimension must be {self.model.dag.nz}, current input dimension X.size()[-1]"
            val = BONSAI_Confidence_Alternator(X, model=self.model, beta=self.beta, maximize=self.maximize, fixed_variable = self.fixed_variable)
            val = val.forward_lcb()

        return val


class BONSAI_Confidence_Alternator():
    def __init__(self,
                 F: Tensor, # The free variable
                 model: Model,
                 beta: Tensor,
                 maximize: bool = True,
                 fixed_variable = None, 
                 ) -> None:
        """
        This is a wrapper that uses alternating bounds
        """

        # Obtain the X values
        self.model = model
        
        self.nx = self.model.dag.nx
        self.nw = self.model.dag.nw
        self.nz = self.model.dag.nz
        self.W_list = self.model.dag.w_combinations
        self.n_nodes = self.model.dag.n_nodes
        self.beta = beta
        
        if self.n_nodes > 5:
            self.Neta = 2**(self.n_nodes)
        else:
            self.Neta = 300
            
        self.fixed_variable = fixed_variable
        self.maximize = maximize
        
        if maximize: # for max min max ucb
            self.Z = F
            self.Nz = F.size()[0]
            if self.W_list is None:
                self.Nw = 50
            else:
                self.Nw = self.W_list.size()[0]
            
        else: # for max max -lcb
            self.W = F  
            self.Nw = F.size()[0]

    def forward_ucb(self):
        # Create Xe corresponds to z + w + eta
        X = torch.empty(self.Nz*self.Nw*self.Neta, self.nz + self.nw)

        #torch.manual_seed(10000)
        # Insert the testing points:
        X[..., self.model.dag.design_input_indices] = self.Z.squeeze(-2).repeat_interleave(self.Nw*self.Neta, dim=0)
        soboleng_w = SobolEngine(dimension= self.nw, seed = 10000)
        if self.W_list is None:            
            X[..., self.model.dag.uncertain_input_indices] = soboleng_w.draw(self.Nw, dtype = torch.double).repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)
        #X[..., self.model.dag.uncertain_input_indices] = torch.rand(self.Nw, self.nw).repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)
        else:
            X[..., self.model.dag.uncertain_input_indices] = self.W_list.repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)            
        
        
        # Create the inter-model calibertaion term
        if self.n_nodes > 5:           
            eta = get_nodes_coeffs(n_nodes = self.model.n_nodes).repeat(self.Nw*self.Nz, 1)             
        else:
            soboleng_eta = SobolEngine(dimension= self.n_nodes, seed = 10000)
            eta = (soboleng_eta.draw(self.Neta, dtype = torch.double) * 2 - 1).repeat(self.Nw*self.Nz, 1)        
        
        #eta = (torch.rand(self.Neta, self.n_nodes) * 2 - 1).repeat(self.Nw*self.Nz, 1)
        
        posterior = self.model.posterior(X)
        ucb_vals = posterior.Bonsai_UCB(
            eta=eta, maximize=self.maximize, beta=self.beta)
        # Assume that the final node is the output of the torch tensor
        ucb_vals = self.model.dag.objective_function(ucb_vals)

        # Create an empty tensor of meshgrids of size Nz X Nw X Neta
        ucb_mesh = torch.empty(self.Nz, self.Nw, self.Neta)

        for i in range(self.Nz):
            ucb_mesh[i] = ucb_vals[i*self.Nw*self.Neta:(i + 1)*self.Nw*self.Neta].reshape(self.Nw,self.Neta)
            
        objective = smooth_amin(smooth_amax(ucb_mesh, dim = -1), dim = -1)

        return objective
    
        
    def forward_lcb(self):
        """
        Maximize over w and eta simultaneously

        Returns
        -------
        int
            DESCRIPTION.

        """
        X = torch.empty(self.Nw, self.nz + self.nw)
        X[..., self.model.dag.design_input_indices] = self.fixed_variable.repeat_interleave(self.Nw, dim=0)
        X[..., self.model.dag.uncertain_input_indices] = self.W.squeeze(1)[:,0:self.nw]
        
        eta = self.W.squeeze(1)[:,self.nw:self.nw+self.n_nodes]*2 - 1 
        
        posterior = self.model.posterior(X)
        lcb_vals = posterior.Bonsai_UCB(
            eta=eta, maximize=self.maximize, beta=self.beta)
        
        # Assume that the final node is the output of the torch tensor
        objective = self.model.dag.objective_function(lcb_vals)           
        
        return objective.squeeze(-2).squeeze(-1)

    def forward_lcb_ver2(self):
        """
        Uses max over w and soft max over eta
        Don't remember why this was done exactly...
        Returns
        -------
        objective : TYPE
            DESCRIPTION.

        """
        X = torch.empty(self.Nw*self.Neta, self.nz + self.nw)
        # Insert the testing points:
        X[..., self.model.dag.design_input_indices] = self.fixed_variable.repeat_interleave(self.Nw*self.Neta, dim=0)
        X[..., self.model.dag.uncertain_input_indices] = self.W.squeeze(-2).repeat_interleave(self.Neta,dim = 0)
        
        #torch.manual_seed(10000)
        soboleng_eta = SobolEngine(dimension= self.n_nodes, seed = 10000)
        eta = (soboleng_eta.draw(self.Neta, dtype = torch.double) * 2 - 1).repeat(self.Nw, 1)
        #eta = (torch.rand(self.Neta, self.n_nodes) * 2 - 1).repeat(self.Nw, 1)
        
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
    



################################################################################
################ Adversarily robust bayesian optimization ######################
################################################################################
 
class ARBO_UCB(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        input_indices: list,
        maximize: bool = True,        
        fixed_variable = None,
        w_combinations = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.fixed_variable = fixed_variable
        self.design_input_indices = input_indices[0]
        self.uncertain_input_indices = input_indices[1]
        self.w_combinations = w_combinations

        # self.n = 1

    @t_batch_mode_transform(expected_q=1)
    def forward(self, Xi: Tensor) -> Tensor:
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
        beta = self.beta.to(Xi)
        nz = len(self.design_input_indices)
        nw = len(self.uncertain_input_indices)
        
        if self.maximize:
            Nz = Xi.size()[0]
            #torch.manual_seed(10000)
            #nz
            
            if self.w_combinations is None:
                Nw = 100
            else:
                Nw = self.w_combinations.size()[0]
            X = torch.empty(Nz*Nw, nz + nw)
            X[..., self.design_input_indices] = Xi.squeeze(-2).repeat_interleave(Nw, dim=0)         
            
            
            if self.w_combinations is None:
                soboleng_w = SobolEngine(dimension= nw, seed = 10000)
                X[..., self.uncertain_input_indices] = soboleng_w.draw(Nw, dtype = torch.double).repeat(Nz,1)   
            else:
                X[..., self.uncertain_input_indices] = self.w_combinations.repeat(Nz,1)
            #X[..., self.uncertain_input_indices] = torch.rand(Nw, nw).repeat(Nz,1)           
            
            posterior = self.model.posterior(X)            
            mean = posterior.mean
            std = posterior.variance.sqrt()
            # Upper confidence bounds
            ucb = mean + beta*std
            
            ucb_mesh = torch.empty(Nz,Nw)
            
            for i in range(Nz):
                ucb_mesh[i] = ucb[i*Nw:(i+1)*Nw].reshape(Nw)
            
            objective = smooth_amin(ucb_mesh, dim = -1)
            
        else:
            if self.fixed_variable is None:
                print('Fixed variable needs to be specified if maximize == "False"')
                sys.exit()
            Nw = Xi.size()[0]
            X = torch.empty(Nw, nz + nw)
            X[..., self.design_input_indices] = self.fixed_variable.repeat_interleave(Nw, dim=0)
            X[..., self.uncertain_input_indices] = Xi.squeeze(-2)
            
            posterior = self.model.posterior(X)            
            mean = posterior.mean
            std = posterior.variance.sqrt()
            
            ucb =  -1*mean + beta*std          
            
            objective = ucb.squeeze(-2).squeeze(-1)          

        return objective
 
    
 
############################################################################
###ARBONS - Adversarilly robust bayesian optimization for Network Systems###
############################################################################
 
    
    
class ARBONS_Acquisition(AnalyticAcquisitionFunction):
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
        self.fixed_variable = fixed_variable
        self.design_input_indices = self.model.dag.design_input_indices
        self.uncertain_input_indices = self.model.dag.uncertain_input_indices
        self.w_combinations = self.model.dag.w_combinations

        # self.n = 1

    @t_batch_mode_transform(expected_q=1)
    def forward(self, Xi: Tensor) -> Tensor:
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
        beta = self.beta.to(Xi)
        nz = len(self.design_input_indices)
        nw = len(self.uncertain_input_indices)
        
        if self.maximize:
            Nz = Xi.size()[0]
            #torch.manual_seed(10000)
            #nz
            
            if self.w_combinations is None:
                Nw = 100
            else:
                Nw = self.w_combinations.size()[0]
            X = torch.empty(Nz*Nw, nz + nw)
            X[..., self.design_input_indices] = Xi.squeeze(-2).repeat_interleave(Nw, dim=0)         
            
            
            if self.w_combinations is None:
                soboleng_w = SobolEngine(dimension= nw, seed = 10000)
                X[..., self.uncertain_input_indices] = soboleng_w.draw(Nw, dtype = torch.double).repeat(Nz,1)   
            else:
                X[..., self.uncertain_input_indices] = self.w_combinations.repeat(Nz,1)
            #X[..., self.uncertain_input_indices] = torch.rand(Nw, nw).repeat(Nz,1)           
            
            posterior = self.model.posterior(X)            
            mean, std = posterior.mean_sigma
            # Upper confidence bounds
            ucb = mean + beta*std
            ucb = self.model.dag.objective_function(ucb)
            
            ucb_mesh = torch.empty(Nz,Nw)
            
            for i in range(Nz):
                ucb_mesh[i] = ucb[i*Nw:(i+1)*Nw].reshape(Nw)
            
            objective = smooth_amin(ucb_mesh, dim = -1)
            
        else:
            if self.fixed_variable is None:
                print('Fixed variable needs to be specified if maximize == "False"')
                sys.exit()
            Nw = Xi.size()[0]
            X = torch.empty(Nw, nz + nw)
            X[..., self.design_input_indices] = self.fixed_variable.repeat_interleave(Nw, dim=0)
            X[..., self.uncertain_input_indices] = Xi.squeeze(-2)
            
            posterior = self.model.posterior(X)            
            mean, std = posterior.mean_sigma            
            ucb =  -1*mean + beta*std  
            ucb = self.model.dag.objective_function(ucb)                    
            objective = ucb.squeeze(-2).squeeze(-1)          

        return objective
    
    
#############################################################################
############ BONSAI-NAB: BONSAI + No Alternating Bounds  ####################
#############################################################################


class BONSAINAB_Acquisition(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        maximize: bool = True, # This is just used to differentiate between z_star and w_star
        fixed_variable = None,
        find_z_star: bool = True
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.n_nodes = self.model.dag.n_nodes
        #self.active_input_indices = self.model.dag.active_input_indices
        self.uncertain_indices = self.model.dag.uncertain_input_indices
        self.design_indices = self.model.dag.design_input_indices    
        
        self.nx = self.model.dag.nx
        self.nw = self.model.dag.nw
        self.nz = self.model.dag.nz
        self.W_list = self.model.dag.w_combinations
        self.n_nodes = self.model.dag.n_nodes
        self.beta = beta
        self.Neta = 200
        self.fixed_variable = fixed_variable  
        self.find_z_star = find_z_star
        

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
        if self.find_z_star:
            self.Z = X
            self.Nz = X.size()[0]
            if self.W_list is None:
                self.Nw = 50
            else:
                self.Nw = self.W_list.size()[0]
                
            # Create Xe corresponds to z + w + eta
            X = torch.empty(self.Nz*self.Nw*self.Neta, self.nz + self.nw)
    
            #torch.manual_seed(10000)
            # Insert the testing points:
            X[..., self.model.dag.design_input_indices] = self.Z.squeeze(-2).repeat_interleave(self.Nw*self.Neta, dim=0)
            soboleng_w = SobolEngine(dimension= self.nw, seed = 10000)
            if self.W_list is None:  
                w_save = soboleng_w.draw(self.Nw, dtype = torch.double)
                X[..., self.model.dag.uncertain_input_indices] = w_save.repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)
            #X[..., self.model.dag.uncertain_input_indices] = torch.rand(self.Nw, self.nw).repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)
            else:
                w_save = self.W_list
                X[..., self.model.dag.uncertain_input_indices] = w_save.repeat(self.Nz, 1).repeat_interleave(self.Neta, dim=0)            
            
            
            # Create the inter-model calibertaion term
            soboleng_eta = SobolEngine(dimension= self.n_nodes, seed = 10000)
            eta_save = soboleng_eta.draw(self.Neta, dtype = torch.double) * 2 - 1
            eta = eta_save.repeat(self.Nw*self.Nz, 1)
            #eta = (torch.rand(self.Neta, self.n_nodes) * 2 - 1).repeat(self.Nw*self.Nz, 1)
            
            posterior = self.model.posterior(X)
            ucb_vals = posterior.Bonsai_UCB(
                eta=eta, maximize=True, beta=self.beta)
            # Assume that the final node is the output of the torch tensor
            ucb_vals = self.model.dag.objective_function(ucb_vals)
    
            # Create an empty tensor of meshgrids of size Nz X Nw X Neta
            ucb_mesh = torch.empty(self.Nz, self.Nw, self.Neta)
    
            for i in range(self.Nz):
                ucb_mesh[i] = ucb_vals[i*self.Nw*self.Neta:(i + 1)*self.Nw*self.Neta].reshape(self.Nw,self.Neta)
            
            objective = smooth_amin(smooth_amax(ucb_mesh, dim = -1), dim = -1)
               
            
            eta_nonsmooth = ucb_mesh.max(-1)    
            eta_nonsmooth_vals= eta_nonsmooth.values
            eta_nonsmooth_indices = eta_nonsmooth.indices
            
            self.eta_vals = eta_save[eta_nonsmooth_indices]
            
            # w_nonsmooth = eta_nonsmooth_vals.max(-1)
            # w_nonsmooth_vals= w_nonsmooth.values
            # w_nonsmooth_indices = w_nonsmooth.indices
            
            # self.w_vals = w_save[w_nonsmooth_indices]
        else:
            self.W = X
            [z_star, eta_star] = self.fixed_variable
            self.Nw = self.W.size()[0]
            self.Nz = 1
            self.Neta = eta_star.size()[0]
            
            X = torch.empty(self.Nz*self.Nw*self.Neta, self.nz + self.nw)
            X[..., self.model.dag.design_input_indices] = z_star.repeat_interleave(self.Nw*self.Neta, dim=0)
            X[..., self.model.dag.uncertain_input_indices] = self.W.squeeze(-1).repeat_interleave(self.Neta, dim=0)
            
            eta = eta_star.repeat(self.Nw, 1)
            
            posterior = self.model.posterior(X)
            ucb_vals = posterior.Bonsai_UCB(
                eta=eta, maximize=True, beta=self.beta)
            # Assume that the final node is the output of the torch tensor
            ucb_vals = self.model.dag.objective_function(ucb_vals)
            
            objective = torch.empty(self.Nw)
            
            for i in range(self.Nw):               
                objective[i] = -1*ucb_vals[i*self.Neta:(i + 1)*self.Neta].min()       

        return objective