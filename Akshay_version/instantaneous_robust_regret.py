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
import time

case_study = ['classifier']
case = case_study[0]

# For plotting the 
T_val = [5*(i+1) for i in range(20)]

function_network, g, nominal_w = function_network_examples(case, algorithm_name= 'Recommendor')

with open('ARBO_'+case+'_recommended.pickle', 'rb') as handle:
    data1 = pickle.load(handle)
    
with open('BOFN_'+case+'_recommended.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
    
with open('BONSAI_'+case+'_recommended.pickle', 'rb') as handle:
    data3 = pickle.load(handle)
    
with open('Random_'+case+'_recommended.pickle', 'rb') as handle:
    data4 = pickle.load(handle)
    
with open('VBO_'+case+'_recommended.pickle', 'rb') as handle:
    data5 = pickle.load(handle)
    

data = {}
data['ARBO'] = data1
data['VBO'] = data5
data['BOFN'] = data2
data['BONSAI'] = data3
data['Random'] = data4


val = {}


for algo in data:
    val[algo] = torch.zeros(30,20) # Here 20 represents the steps for iteration 5, 10,... 100
    print(algo)   

    if g.nw != 0:   
        maxmin_val = torch.empty(g.w_combinations.size()[0],1)
        all_vals = 0 
        for i in range(30):
            print('Repeat Number', i)
            j = 0
            for t in T_val:
                t1 = time.time()
                print('Iteration value', t)
                X_empty = torch.empty(g.w_combinations.size()[0], g.nx)
                Z_new = data[algo][i,t]['Z'].repeat(g.w_combinations.size()[0],1)
                
                X_empty[..., g.design_input_indices] = Z_new
                X_empty[..., g.uncertain_input_indices] = g.w_combinations
                
                if case == 'classifier':
                    all_vals = g.objective_function(function_network(X_empty, test_mode = True))                    
                else:                    
                    all_vals = g.objective_function(function_network(X_empty))
                
                worst_case = all_vals.min()
                val[algo][i,j] = worst_case
                j += 1
                t2 = time.time()
                
                print('worst-case value =', worst_case)
                print('time for this evaluation',t2 - t1)
    
    else:
       for i in range(30):
           j = 0
           for t in T_val:           
               val[algo][i,j] = data[algo][i,t]['Y']
               j += 1   
        
    
    
    
    

# Get the maximum + minimum vals
max_val = []
min_val = []
for algo in data:
    max_val.append(val[algo].max())
    min_val.append(val[algo].min())

max_val = max(max_val)
min_val = min(min_val)

# Plot the robust regret:

color = ['blue', 'green','red','black', 'orange']
j = 0

for algo in data:    

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


with open(case+'_plot_final.pickle', 'wb') as handle:
    pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)