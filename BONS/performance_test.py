#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:39:13 2024

@author: kudva.7
"""
import pickle
import torch
from Objective_FN import function_network_examples
from algorithms import BOFN, BONS


examples = ['dropwave', 'toy_problem']
example = examples[0]

acq_fun = 'qUCB'

print('Running ' + example)

# Extract the example and dag structure

function_network, g = function_network_examples(example = example)

#################################################################################

input_dim = g.nx  
n_outs = g.n_nodes

# Start the modeling procedure
Ninit = 2*input_dim + 1
T = 10


Nrepeats = 1

BOFN_qlogEI = {}
BONS_val = {}


for n in range(Nrepeats):
    j = n + 1     
    print('Current repeat', j)  
    torch.manual_seed(seed = (j + 1)*2000)
    x_init = torch.rand(Ninit, g.nx)
    y_init = function_network(x_init)   
    
    
    # print('Running BOFN with ' + acq_fun)
    # val = BOFN( x_init, y_init, g, objective = function_network, T = T, acq_type = acq_fun, q = 3)             
    # BOFN_qlogEI[n] = val
    
    val = BONS( x_init, y_init, g, objective = function_network, T = T, q = 3)             
    BONS_val[n] = val 
    
    


drop_wave = {}
drop_wave['BOFN_qlogEI'] = BOFN_qlogEI


  
# with open('drop_wave.pickle', 'wb') as handle:
#     pickle.dump(drop_wave, handle, protocol=pickle.HIGHEST_PROTOCOL)        








