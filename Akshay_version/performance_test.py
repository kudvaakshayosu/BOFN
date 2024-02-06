#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:24:00 2024

@author: kudva.7
"""

import torch
from ObjectiveFN import function_network_examples
from robust_algorithms import BONSAI, ARBO
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt

example_list = ['concave_two_dim', 'non_concave_two_dim']
algorithm_list = ['BONSAI', 'ARBO']


example = example_list[1]

function_network, g = function_network_examples(example)
nz = g.nz
nw = g.nw
input_dim = g.nx  
n_outs = g.n_nodes

# Start the modeling procedure
Ninit = 5

BONSAI_data = {}
ARBO_data = {}

Nrepeats = 100

for n in range(Nrepeats):
    j = n + 1     
    print('Current repeat', j)
    torch.manual_seed((j+1)*2000)
    #soboleng_x = SobolEngine(dimension= input_dim, seed = (j+1)*2000)
    #x_init = soboleng_x.draw(Ninit, dtype = torch.double)
    x_init = torch.rand(Ninit,input_dim)
    y_init = torch.zeros(Ninit,n_outs)
    
    for i in range(Ninit):
        y_init[i] = function_network(x_init[i])
    
    if 'BONSAI' in algorithm_list:
        print('Running BONSAI')
        val = BONSAI( x_init, y_init, g, objective = function_network, T = 10, beta = torch.tensor(2))
        BONSAI_data[n] = val
        
    if 'ARBO' in algorithm_list:
        print('Running ARBO')
        val = ARBO(x_init, y_init, g, objective = function_network, T = 10, beta = torch.tensor(2))
        ARBO_data[n] = val
