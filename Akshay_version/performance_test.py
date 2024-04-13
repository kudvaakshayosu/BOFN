#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:24:00 2024

@author: kudva.7
"""

import torch
from ObjectiveFN import function_network_examples
from robust_algorithms import BONSAI, ARBO, ARBONS, BOFN_nominal_mode
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
import pickle
from utils import generate_initial_data
import copy

example_list = ['synthetic_fun1_discrete', 'covid_testing', 'polynomial', 'robot', 'classifier', 'cliff', 'HeatX', 'sine']
algorithm_list = ['BOFN'] #['BONSAI', 'ARBO', 'BOFN_nominal', 'Random', 'VBO']


example = example_list[7]

function_network, g, nominal_w = function_network_examples(example, algorithm_name = algorithm_list[0])
nz = g.nz
nw = g.nw
input_dim = g.nx  
n_outs = g.n_nodes

# Start the modeling procedure
Ninit = 2*(input_dim + 1)
T = 100


data = {}
Nrepeats = 30

for n in range(Nrepeats):
    j = n + 1     
    print('Current repeat', j)   
    if 'Random' in algorithm_list:
        x_init, y_init = generate_initial_data( g = g, function_network = function_network, Ninit = Ninit + T, seed = (j+1)*2000, get_y_vals= True)
        
        val = {'X': x_init, 'Y': y_init,'Ninit': Ninit, 'T': T}
        data[n] = val
        
    elif 'BOFN' in algorithm_list or 'VBO' in algorithm_list:
        x_init, y_init = generate_initial_data( g = g, function_network = function_network, Ninit = Ninit, seed = (j+1)*2000, get_y_vals= False)
    
    else:
        x_init, y_init = generate_initial_data( g = g, function_network = function_network, Ninit = Ninit, seed = (j+1)*2000, get_y_vals= True)
        
    if 'ARBO' in algorithm_list:
        print('Running ARBO')
        val = ARBO(x_init, y_init, g, objective = function_network, T = T, beta = torch.tensor(3))
        data[n] = val
        
    if 'ARBONS' in algorithm_list:
        print('Running ARBONS')
        val = ARBONS(x_init, y_init, g, objective = function_network, T = T, beta = torch.tensor(2))
        data[n] = val
        
    if 'BOFN' in algorithm_list or 'VBO' in algorithm_list:
        print('Running BOFN - nominal mode')
        if example == 'robot':
            if 'BOFN' in algorithm_list:
                val = BOFN_nominal_mode(x_init, y_init, g, objective = function_network , T = T, nominal_w = nominal_w) # Also keep get_g_vals true in this case
                data[n] = val
            if 'VBO' in algorithm_list:
                val = BOFN_nominal_mode(x_init, y_init, g, objective = function_network , T = T, nominal_w = nominal_w, graph_structure= False) # Also keep get_g_vals true in this case
                data[n] = val 
            
        else:   
            # Note: Keep get_g_vals to False since we are setting a nominal value specific to the problem
            x_init2 = copy.deepcopy(x_init)
            x_init2[..., g.uncertain_input_indices] = nominal_w        
            x_init2, y_init2 = generate_initial_data( g = g, function_network = function_network, Ninit = Ninit, x_init = x_init2)
            if 'BOFN' in algorithm_list:
                val = BOFN_nominal_mode(x_init2, y_init2, g, objective = function_network , T = T, nominal_w = nominal_w)
                data[n] = val
            if 'VBO' in algorithm_list:
                val = BOFN_nominal_mode(x_init2, y_init2, g, objective = function_network , T = T, nominal_w = nominal_w, graph_structure= False)
                data[n] = val           
        
    if 'BONSAI' in algorithm_list:
        print('Running BONSAI')               
        val = BONSAI( x_init, y_init, g, objective = function_network, T = T)    
        data[n] = val
        
    


with open(algorithm_list[0]+'_'+example+'.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)