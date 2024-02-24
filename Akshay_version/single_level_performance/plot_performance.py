#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:05:49 2024

@author: kudva.7
"""

import pickle
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


test_cases = ['drop_wave', 'config_one', 'config_one_noise','config_three', 'drop_wave_noise']

# Select the index of the case study you want to plot
test_case = test_cases[4]



with open( test_case + '.pickle', 'rb') as handle:
    value = pickle.load(handle)

plot_time = True
plot_performance = True

if plot_time:
    
    # Generate empty list for time and algorithms
    time_list = []
    algo_list = []
    
    # Extract time values of each algorithm
    for val in value:
        algo_list.append(val)
        Nrepeats = len(value[val])
        time_total = 0
        
        for i in range(Nrepeats):
            time_val = torch.tensor(value[val][i]['Time'])
            time_total += time_val
            
        time_total = time_total/Nrepeats
        time_list.append(float(time_total.sum()))
    
    # Plotting the bar with time
    plt.bar(algo_list, time_list)
    
    # Adding labels and title
    plt.xlabel('Algorithm')
    plt.ylabel('Average Total Time over 10 repeats' )
    plt.title('Time comparison between baselines')
    plt.xticks(rotation=90)
    
    # Displaying the plot
    plt.show()


if plot_performance:
    obj_list = []
    algo_list = []
    
    # Exract values of each algorithm
    for val in value:
        algo_list.append(val)
        Nrepeats = len(value[val])
        Ninit = value[val][0]['Ninit']
        T = value[val][0]['T']
        obj_vals = torch.empty(Nrepeats, T)
        
        # Extract the valid values
        for i in range(Nrepeats):
            obj_vals[i,:] = value[val][i]['Y'][Ninit:,-1]
            
        obj_list.append(obj_vals)
   
    simple_regret_list = []
    
    # Generate simple reward
    for obj_val in obj_list:
        simple_regret = torch.empty(obj_val.size())
        
        for i in range(T):
            simple_regret[:,i] = obj_val[:,0:i+1].max(dim = 1).values
            
        simple_regret_list.append(simple_regret)
        
    # Plot everything for each algorithm
    means = [torch.mean(element, dim = 0) for element in simple_regret_list ]
    std = [torch.std(element, dim = 0) for element in simple_regret_list ]
    
    colors = ['blue', 'yellow', 'orange', 'green', 'magenta', 'brown', 'red' ]
    i = 0
    x_val = [i for i in range(1, 1 + T)]
    
    
    
    for algo in algo_list:
        plt.plot(x_val,means[i], label = algo, color = colors[i])
        plt.fill_between(x_val,means[i] - 1.96*std[i]/torch.sqrt(torch.tensor(Nrepeats)), means[i] + 1.96*std[i]/torch.sqrt(torch.tensor(Nrepeats)), alpha=0.2, color=colors[i])
        i += 1
        
    plt.xlabel('Iteration,t')
    plt.xticks(x_val)
    plt.ylabel('Reward')
    plt.title('Performance for ' + test_case + 'case study')
    plt.legend()
    plt.grid()
    plt.show()



