#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:42:57 2024

@author: kudva.7
"""

import pickle 
import torch
import matplotlib.pyplot as plt

case = ['polynomial', 'robot', 'classifier','cliff', 'HeatX']
case_study = [case[4]]
case = case_study[0]

# For plotting the 
T_val = [5*(i+1) for i in range(20)]

with open(case+'_plot_final.pickle', 'rb') as handle:
    val = pickle.load(handle)

# Get the maximum + minimum vals
max_val = []
min_val = []

for algo in val:
    max_val.append(val[algo].max())
    min_val.append(val[algo].min())

max_val = max(max_val)
min_val = min(min_val)

# Plot the robust regret:

color = ['blue', 'green','red', 'black', 'magenta']
j = 0

for algo in val:    
    #val1 = (val[algo] - min_val)/(max_val - min_val)
    val1 = val[algo]
    means = [torch.mean(element, dim = 0) for element in val1.T]
    std = [torch.std(element, dim = 0) for element in val1.T]
    
    plt.plot(T_val,means, label = algo, color = color[j])
    plt.fill_between(T_val,torch.tensor(means) - 1.96*torch.tensor(std)/torch.sqrt(torch.tensor(30)), torch.tensor(means) + 1.96*torch.tensor(std)/torch.sqrt(torch.tensor(30)), alpha=0.2, color= color[j])

    j += 1    
        
plt.xlabel('Iteration,t')
plt.xticks(T_val)
plt.ylabel('Instantaneous Robust Reward Percentage')
plt.title('Performance for ' + case + ' case study')
plt.xlim([5,100])
plt.legend()
plt.grid()
plt.show()    