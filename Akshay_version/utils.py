#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:17:55 2024

@author: kudva.7
"""

import torch
from torch import Tensor
from graph_utils import Graph
from typing import Callable
import copy



# Use to round the nearest values to the discrete units
def round_to_nearest_set(input_tensor, sets_of_values):
    """
    Used to round a value to the nearest set of discrete values. Used for specifying an uncertainty value
    
    Parameters
    ----------
    input_tensor : Tensor
        This is the value that needs rounding to
    sets_of_values : List
        This is the set that contains the dicerete value of uncertainties.
        Needs to be a list of lists (corresponding to the number of dimensionsof uncertain variables)
        
    Returns
    -------
    Tensor
        Returns a set of rounded torch tensors
    """
    rounded_values = []
    for tensor in input_tensor:
        rounded_tensor = []
        for value, set_of_values in zip(tensor, sets_of_values):
            rounded_value = min(set_of_values, key=lambda x: abs(x - value))
            rounded_tensor.append(rounded_value)
        rounded_values.append(rounded_tensor)
    return torch.tensor(rounded_values).type(torch.double)



def generate_initial_data(g: Graph,
                          function_network: Callable, 
                          seed: int = 2000,
                          Ninit: int = 10, 
                          x_init: Tensor = None,
                          get_y_vals = True
                          ):
    
    """
    Used to generate the initial values with or without discrete sampling
    with different values of initial seeds
    
    Parameters
    ---------
    g: Graph
        The graph contains all the information regarding the function network 
        
    function network: Callable
        The Objective function value
    
    Ninit: int
        Number of initial values
    
    seed: int
        Seed number 
    
    x_init: Tensor
        Values of the torch tensor incase some initial search values are provided
    
   Returns
   -------
    x_init: Initial values of decision variables
    y_init: Function evaluations at the corresponding x_init values    
    """

    
    # Identify the type of problem
   
    if g.nw == 0:   
        if g.w_combinations is not None:      
            nw = g.w_combinations.size()[1]
        else:
            nw = 0
        if nw == 0:
            print('This is a single level problem')
    else:
        nw = g.nw
        print('Uncertain variables are continuous')
     
    # Generate random initialization of design based on the seed
    input_dim = g.nz + nw 
    n_outs = g.n_nodes
    
    torch.manual_seed(seed)
    
    if x_init is None: # this is if some default designs are already provided
        x_init = torch.rand(Ninit,input_dim)
    
    ##########################################################
    if g.w_combinations is None:
        x_init2 = copy.deepcopy(x_init)
        
    else: # Use these designs to generate values
        x_init2 = copy.deepcopy(x_init)    
        if len(g.uncertain_input_indices) > 0:
            rounded_vals = round_to_nearest_set(x_init2[...,g.uncertain_input_indices], g.w_sets)
            x_init[...,g.uncertain_input_indices] = rounded_vals 
        else:
            uncertain_input_indices = [i for i in range(g.nz,g.nz + nw)]
            rounded_vals = round_to_nearest_set(x_init2[...,uncertain_input_indices], g.w_sets)
            x_init[...,uncertain_input_indices] = rounded_vals 
            
           
    
    y_init = torch.zeros(Ninit,n_outs)
    
    if not get_y_vals:
        return x_init, y_init
    
    
    for i in range(Ninit):
        y_init[i] = function_network(copy.deepcopy(x_init[i]))  
    
    return x_init, y_init





