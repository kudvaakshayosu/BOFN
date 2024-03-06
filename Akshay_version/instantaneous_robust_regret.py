#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:15:12 2024
Plot robust performance
@author: kudva.7
"""

import pickle 
import torch
import matplotlib.pyplot as plt
from ObjectiveFN import function_network_examples
import matplotlib.pyplot as plt



# For plotting the 
T_val = [5*(i+1) for i in range(19)]

function_network, g, nominal_w = function_network_examples('polynomial')

with open('ARBO_polynomial_recommended.pickle', 'rb') as handle:
    data1 = pickle.load(handle)
    
with open('BOFN_polynomial_recommended.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
    
with open('BONSAI_polynomial_recommended.pickle', 'rb') as handle:
    data3 = pickle.load(handle)
    
data = {}
data['ARBO'] = data1
data['BOFN'] = data2
data['BONSAI'] = data3

val = {}

for algo in data:
    val[algo] = torch.zeros(30,19) 
    all_vals = 0   
    
    maxmin_val = torch.empty(g.w_combinations.size()[0],1)
      
    for i in range(30):
        j = 0
        for t in T_val:
            X_empty = torch.empty(g.w_combinations.size()[0], g.nx)
            Z_new = data[algo][i,t]['Z'].repeat(g.w_combinations.size()[0],1)
            
            X_empty[..., g.design_input_indices] = Z_new
            X_empty[..., g.uncertain_input_indices] = g.w_combinations
            
            all_vals = g.objective_function(function_network(X_empty))
            
            val[algo][i,j] = all_vals.min()
            j += 1




# Plot the robust regret:

color = ['blue', 'green', 'red']
j = 0

for algo in data:    

    val1 = val[algo]    
    means = [torch.mean(element, dim = 0) for element in val1.T]
    std = [torch.std(element, dim = 0) for element in val1.T]
    
    plt.plot(T_val,means, label = algo, color = color[j])
    plt.fill_between(T_val,torch.tensor(means) - 1.96*torch.tensor(std)/torch.sqrt(torch.tensor(30)), torch.tensor(means) + 1.96*torch.tensor(std)/torch.sqrt(torch.tensor(30)), alpha=0.2, color= color[j])

    j += 1    
        
plt.xlabel('Iteration,t')
plt.xticks(T_val)
plt.ylabel('Instantaneous Robust Reward')
plt.title('Performance for polynomial case study')
plt.xlim([5,95])
plt.legend()
plt.grid()
plt.show()    