#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:15:59 2024
This code is used for recommendation of ARBO and or BONSAI
@author: kudva.7
"""

import pickle 
import torch
import matplotlib.pyplot as plt
from robust_algorithms import BONSAI_Recommendor, ARBO_Recommendor_final, BOFN_Recommendor_final, Mean_Recommendor_final
from ObjectiveFN import function_network_examples


function_network, g, nominal_w = function_network_examples('robot', algorithm_name= 'Recommendor')


recommendor = 'by_iteration_not'

T_val = [5*(i+1) for i in range(20)]
#T_val = [100]

with open('BONSAI_robot.pickle', 'rb') as handle:
    data = pickle.load(handle)

BONSAI_recommender_data = {}    

if recommendor == 'by_iteration':
    # This is for recommending design by-iteration
    for i in data:
        BONSAI_recommender_data[i] = BONSAI_Recommendor(data = data[i], g = g)    
else:
    # This is for recommending a final set of best design based on all available data
    for i in data:
        for T in T_val:
            BONSAI_recommender_data[i,T] = Mean_Recommendor_final(data = data[i], g = g, T = T ) 
    
# with open('BONSAI_robot_recommended.pickle', 'wb') as handle:
#     pickle.dump(BONSAI_recommender_data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
