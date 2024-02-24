#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:39:13 2024

@author: kudva.7
"""
import pickle
import torch
from utils import generate_initial_data
from single_level_Objective import function_network_examples
from single_level_algorithms import BOFN, BayesOpt, BONS


examples = ['dropwave', 'config_one', 'config_two', 'config_three']

algorithms = ['BOFN', 'BO', 'BONS',]
acq_type = ['qEI','qlogEI','qUCB']

algorithm_list = algorithms


example = examples[0]
print('Running ' + example)

# Extract the example and dag structure

function_network, g = function_network_examples(example = example)

#################################################################################

function_network, g = function_network_examples(example)
input_dim = g.nx  
n_outs = g.n_nodes

# Start the modeling procedure
Ninit = 2*input_dim + 1
T = 40

BOFN_qEI = {}
BOFN_qlogEI = {}
BOFN_qUCB = {}

BO_qEI = {}
BO_qlogEI = {}
BO_qUCB = {}

BONS_UCB = {}

Nrepeats = 20

for n in range(Nrepeats):
    j = n + 1     
    print('Current repeat', j)   
    x_init, y_init = generate_initial_data( g = g, function_network = function_network, Ninit = Ninit, seed = (j+1)*2000)  
      
    for algorithm_name in algorithm_list:
        if algorithm_name == 'BOFN':
            for acq_fun in acq_type:
                print('Running BOFN with ' + acq_fun)
                val = BOFN( x_init, y_init, g, objective = function_network, T = T, acq_type = acq_fun)
            
                if acq_fun == 'qEI':
                    BOFN_qEI[n] = val
                elif acq_fun == 'qlogEI':
                    BOFN_qlogEI[n] = val
                elif acq_fun == 'qUCB':
                    BOFN_qUCB[n] = val      
            
            
        elif algorithm_name == 'BO':
            acq_fun = acq_type[1]
            print('Running BO with ' + acq_fun)
            val = BayesOpt( x_init, y_init, g, objective = function_network, T = T, acq_type = acq_fun)
            
            if acq_fun == 'qEI':
                BO_qEI[n] = val
            elif acq_fun == 'qlogEI':
                BO_qlogEI[n] = val
            elif acq_fun == 'qUCB':
                BO_qUCB[n] = val    
        
        else:
            print('Running BONS')
            val = BONS( x_init, y_init, g, objective = function_network, T = T)        
            BONS_UCB[n] = val
        


drop_wave = {}

drop_wave['BOFN_qEI'] = BOFN_qEI

drop_wave['BOFN_qlogEI'] = BOFN_qlogEI

drop_wave['BOFN_qUCB'] = BOFN_qUCB

#drop_wave['BO_qEI'] = BO_qEI

drop_wave['BO_qlogEI'] = BO_qlogEI

#drop_wave['BO_qUCB'] = BO_qUCB

drop_wave['BONS'] = BONS_UCB


config_three =  drop_wave
  
with open('drop_wave_noise.pickle', 'wb') as handle:
    pickle.dump(drop_wave, handle, protocol=pickle.HIGHEST_PROTOCOL)        








