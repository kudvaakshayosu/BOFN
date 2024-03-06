#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:23:29 2024

@author: kudva.7
"""

import pickle
import torch


# Synthetic function values
fun = lambda x: 100*(torch.sin(4.7/(1+ torch.log(x[:,0] + x[:,1]))) + 2*torch.cos(0.5*x[:,1]))



with open('BONSAI_non_concave_twoD.pickle', 'rb') as handle:
    BONSAI_data = pickle.load(handle)
    
    
with open('ARBO_non_concave_twoD.pickle', 'rb') as handle:
    ARBO_data = pickle.load(handle)
    
w_n = 20
z_min = torch.tensor(4)
z_max = torch.tensor(20)

w_min = -1
w_max = 1

# z_min = torch.tensor(-1)
# z_max = torch.tensor(2)

# w_min = 2
# w_max = 4

w_test = torch.linspace(w_min,w_max,w_n).unsqueeze(1)
data = ARBO_data

Nrepeats = 100
Nsamples = 15
Ninit = 5

# Nrepeats = 100
# Nsamples = 20
# Ninit = 10

fun_val_repeats = torch.zeros(Nsamples, Nrepeats)
time_val_repeats = torch.zeros(Nsamples - Ninit, Nrepeats)

# Extract all the values
for i in range(Nrepeats):
    fun_val = []
    for j in range(Nsamples):
        x = data[i]['X'][j][0].repeat(20,1)*(z_max - z_min) + z_min
        f_test = fun(torch.cat([x,w_test], dim = 1)).min()
        fun_val_repeats[j, i] = f_test
        
    time_vals = torch.tensor(data[i]['T1']) + torch.tensor(data[i]['T2'])      
    time_val_repeats[:,i] = time_vals
        

# ARBO_extracted = {}
# ARBO_extracted['F_min_W'] = fun_val_repeats
# ARBO_extracted['time'] = time_val_repeats
    
# with open('ARBO_non_concave_twoD_extracted.pickle', 'wb') as handle:
#     pickle.dump(ARBO_extracted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
        



