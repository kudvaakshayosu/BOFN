#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:24:00 2024

@author: kudva.7
"""

import torch
from ObjectiveFN import function_network_examples
from robust_algorithms import BONSAI, ARBO, ARBONS, BONSAI_NAB
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
import pickle
from utils import generate_initial_data
import copy

example_list = ['synthetic_fun1','synthetic_fun1_discrete', 'synthetic_fun2', 'synthetic_fun3','covid_testing']
algorithm_list = ['BONSAI_NAB'] #['BONSAI', 'ARBO', 'ARBONS', 'BONSAI_NAB]


example = example_list[1]

function_network, g = function_network_examples(example)
nz = g.nz
nw = g.nw
input_dim = g.nx  
n_outs = g.n_nodes

# Start the modeling procedure
Ninit = 20

BONSAI_data = {}
ARBO_data = {}
ARBONS_data = {}
BONSAI_NAB_data = {}

Nrepeats = 1

for n in range(Nrepeats):
    j = n + 1     
    print('Current repeat', j)   
    x_init, y_init = generate_initial_data( g = g, function_network = function_network, Ninit = Ninit, seed = (j+1)*2000)
    
    if 'BONSAI' in algorithm_list:
        print('Running BONSAI')
        val = BONSAI( x_init, y_init, g, objective = function_network, T = 10, beta = torch.tensor(2))
        BONSAI_data[n] = val
        
    if 'ARBO' in algorithm_list:
        print('Running ARBO')
        val = ARBO(x_init, y_init, g, objective = function_network, T = 10, beta = torch.tensor(2))
        ARBO_data[n] = val
        
    if 'ARBONS' in algorithm_list:
        print('Running ARBONS')
        val = ARBONS(x_init, y_init, g, objective = function_network, T = 10, beta = torch.tensor(2))
        ARBONS_data[n] = val
        
    if 'BONSAI_NAB' in algorithm_list:
        print('Running BONSAI-NAB')
        val = BONSAI_NAB( x_init, y_init, g, objective = function_network, T = 10, beta = torch.tensor(2))
        BONSAI_data[n] = val
        
    


# with open('BONSAI_ARBO_example.pickle', 'wb') as handle:
#     pickle.dump(BONSAI_data, handle, protocol=pickle.HIGHEST_PROTOCOL)