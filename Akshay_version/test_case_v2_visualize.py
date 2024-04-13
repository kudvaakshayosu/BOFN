#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:42:25 2024

@author: kudva.7
"""
import torch
import matplotlib.pyplot as plt
from ObjectiveFN import function_network_examples
from utils import generate_initial_data

plot_synthetic = True
plot_covid_sensitivities = False

if plot_synthetic:

    ####################
    fun = lambda x: 10/(1+ torch.log(x[:,0] + x[:,1]))
    
    #fun = lambda x: 100*(torch.sin(4.7/(1+ torch.log(x[:,0] + x[:,1]))) + 2*torch.cos(0.5*x[:,1]))
    
    # fun = lambda z: (-2*(z[:,0])**6 + 12.2*(z[:,0])**5 - 21.2*(z[:,0])**4 - 6.2*(z[:,0]) + 6.4*(z[:,0])**3 + 4.7*(z[:,0])**2 
    # - (z[:,1])**6 + 11*(z[:,1])**5 - 43.3*(z[:,1])**4 + 10*(z[:,1]) + 74.8*(z[:,1])**3 - 56.9*(z[:,1])**2
    # + 4.1*(z[:,0])*(z[:,1]) + 0.1*((z[:,1])**2)*((z[:,0])**2) - 0.4*((z[:,1])**2)*(z[:,0]) - 0.4*((z[:,0])**2)*(z[:,1]))
    
    LB = torch.tensor([4,-1])
    UB = torch.tensor([20,1])
    ####################
    
    # fun = lambda x: -1*(torch.sin(x[:,0]*x[:,1]) + x[:,1].sqrt()*(x[:,0])**2 - 0.5*x[:,0])
    
    # LB = torch.tensor([-1,2])
    # UB = torch.tensor([2,4])
    
    #######################################
    # LB = torch.tensor([-0.5,-0.5])
    # UB = torch.tensor([3.25,4.25])
        
        
        
    xaxis = torch.arange(LB[0], UB[0], (UB[0]-LB[0])/1000)
    yaxis = torch.arange(LB[1], UB[1], (UB[1]-LB[1])/1000)
    
    # create a mesh from the axis
    x2, y2 = torch.meshgrid(xaxis, yaxis)
    
    # reshape x and y to match the input shape of fun
    xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)
    
    
    results = fun(xy)
    
    results2 = results.reshape(x2.size())
    
    
    inner_min = results2.min(dim = 1)
    
    min_indices = inner_min.indices
    
    robust_index = torch.argmax(inner_min.values)
    
    robust_find = torch.max(inner_min.values)
    robust_pt = x2[robust_index][0]
    worst_case = y2[:,min_indices[robust_index]][0]
    
    ######## Get the best value ###########
    xy_best = xy[results.argmax()]
    
    ####
    fig, ax = plt.subplots(1, 1)
    plt.set_cmap("jet")
    contour_plot = ax.contourf(x2,y2,results2)
    fig.colorbar(contour_plot)
    plt.vlines(robust_pt, LB[1], UB[1], colors = 'black', linestyles = 'dashed')
    
    plt.scatter(xy_best[0], xy_best[1], marker = '*', color = 'white', s = 200)
    
    plt.xlim(LB[0], UB[0])
    plt.ylim(LB[1], UB[1])
    
    
    plt.xlabel('$z_{1}$', fontsize = 20)
    plt.ylabel('$z_{2}$', fontsize = 20)
 
################################################
if plot_covid_sensitivities:
    function_network, g, _ = function_network_examples('covid_testing')
    all_w_combinations = g.w_combinations
    num_combs = all_w_combinations.size()[0]
    
    
    fixed_values = torch.tensor([0.31,0.9593,0.00])
    fixed_values = fixed_values.repeat(num_combs,1)
    
    all_vals = torch.cat((fixed_values, all_w_combinations), dim = 1)
    
    y_out = torch.empty(num_combs,1)
    
    
    for i in range(num_combs):
        print('itertion number', i)
        y_out[i] = g.objective_function(function_network(all_vals[i]))
        
    y_plot = y_out.reshape(4,6)
    
    w1 = torch.tensor(g.w_sets[0])*3 + 12
    w2 = torch.tensor(g.w_sets[1])*0.01 + 0.01
    
    
    w1_mesh, w2_mesh = torch.meshgrid(w1,w2)
    
    fig, ax = plt.subplots(1, 1)
    plt.set_cmap("jet")
    contour_plot = ax.contour(w1_mesh,w2_mesh,y_plot)
    fig.colorbar(contour_plot)
    
    plt.xlabel('Testing time steps')
    plt.ylabel('Prevalence')
    
        
    


