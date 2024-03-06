#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:23:29 2024
This function is used to plot the simple regret and 
robust regret in the initial versions for LogEI

@author: kudva.7
"""

import pickle 
import torch
import matplotlib.pyplot as plt


objective_function = lambda Y: -100 * torch.sum(Y[..., [3*t + 2 for t in range(3)]], dim=-1)

with open('BOFN_nominal_part1.pickle', 'rb') as handle:
    BOFN_data = pickle.load(handle)

Nrepeats = len(BOFN_data) 
Ninit =    BOFN_data[0]['Ninit']
T = BOFN_data[0]['T']

Y_single_level = torch.empty(Nrepeats, T)
Y_max = torch.empty(Nrepeats,T)

for trial in BOFN_data:
    Y_single_level[trial,:] = objective_function(BOFN_data[trial]['Y'])[Ninit:]





for i in range(T):  
    if i == 0:
        Y_max[:,i] = Y_single_level[:,i]
    else:
        Y_max[:,i] = Y_single_level[:,0:i+1].max(dim = 1).values
        
means = torch.mean(Y_max, dim = 0) 
std = torch.std(Y_max, dim = 0)
x_val = [i for i in range(T)]


plt.plot(x_val,means, label = 'LogEI', color = 'blue')
plt.fill_between(x_val,means - 1.96*std/torch.sqrt(torch.tensor(Nrepeats)), means + 1.96*std/torch.sqrt(torch.tensor(Nrepeats)), alpha=0.2, color= 'blue')
       




