#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:52:07 2024

@author: kudva.7
"""

import torch
from torch import Tensor
import matplotlib.pyplot as plt

class Polynomial:
    def __init__(self):
        self.n_nodes = 3
        self.input_dim = 4

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        X1 = X[...,0] + X[...,2]*torch.cos(X[...,3])
        X2 = X[...,1] + X[...,2]*torch.sin(X[...,3])
        
        output[..., 0] = -2*(X1)**6 + 12.2*(X1)**5 - 21.2*(X1)**4 - 6.2*(X1) + 6.4*(X1)**3 + 4.7*(X1)**2 
        output[..., 1] = -1*(X2)**6 + 11*(X2)**5 - 43.3*(X2)**4 + 10*(X2) + 74.8*(X2)**3 - 56.9*(X2)**2
        output[..., 2] = + 4.1*(X1)*(X2) + 0.1*((X2)**2)*((X1)**2) - 0.4*((X2)**2)*(X1) - 0.4*((X1)**2)*(X2)
        
        return output




if __name__ == '__main__':
    """        Hello world!
    """
    import itertools 
    
    sanity_check = False
    visualize_robust = False
    
    torch.set_default_dtype(torch.float64)
    dropwave = Polynomial()
    input_dim = dropwave.input_dim

    fun = lambda z: (-2*(z[:,0])**6 + 12.2*(z[:,0])**5 - 21.2*(z[:,0])**4 - 6.2*(z[:,0]) + 6.4*(z[:,0])**3 + 4.7*(z[:,0])**2 
    - (z[:,1])**6 + 11*(z[:,1])**5 - 43.3*(z[:,1])**4 + 10*(z[:,1]) + 74.8*(z[:,1])**3 - 56.9*(z[:,1])**2
    + 4.1*(z[:,0])*(z[:,1]) + 0.1*((z[:,1])**2)*((z[:,0])**2) - 0.4*((z[:,1])**2)*(z[:,0]) - 0.4*((z[:,0])**2)*(z[:,1]))
    
    LB = torch.tensor([-0.5,-0.5])
    UB = torch.tensor([3.25,4.25])
        
    N = 1000   
        
    xaxis = torch.arange(LB[0], UB[0], (UB[0]-LB[0])/N)
    yaxis = torch.arange(LB[1], UB[1], (UB[1]-LB[1])/N)
    
    # create a mesh from the axis
    x2, y2 = torch.meshgrid(xaxis, yaxis)
    
    # reshape x and y to match the input shape of fun
    xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)
    
    
    results = fun(xy)
    
    results2 = results.reshape(x2.size())
    
    w_set = [[0,0.2,0.4,0.6,1.],[0.0000, 0.100, 0.1250, 0.2000, 0.2500, 0.300, 0.3750, 0.4500, 0.5000, 0.5700, 0.6250, 0.700, 0.7500, 0.8750, 0.9500, 1.0000]]
    
    all_combinations = itertools.product(*w_set)
    tensors = [torch.tensor(combination) for combination in all_combinations]
    
    # Stack the tensors to create the final result
    w_combinations = torch.stack(tensors)
    J = w_combinations.shape[0]
    
    worst_case_vals = torch.empty(results.size())
    
    if visualize_robust:
        for i in range(N**2):
            x_tilde = xy[i].repeat(J,1)
            X_vals = torch.hstack((x_tilde, w_combinations))
            
            worst_case_vals[i] = dropwave.evaluate(X_vals).min()       
            
        results3 = worst_case_vals.reshape(x2.size())     
        xy_robust = xy[worst_case_vals.argmax()]    
            
    
    ############## Generate 
    
    
    ######## Get the best value ###########
    xy_best = xy[results.argmax()]
    
    
    ####
    
    vmin = -1000
    vmax = 10
    
    levels = torch.linspace(vmin,vmax,100)
    
    fig, ax = plt.subplots(1, 1)
    plt.set_cmap("jet")
    
    if visualize_robust:
        contour_plot = ax.contourf(x2,y2,results3, levels = levels)
        plt.scatter(xy_robust[0], xy_robust[1], marker = '*', color = 'black', s = 200, label = 'Nominal Solution')
    else:
        contour_plot = ax.contourf(x2,y2,results2)
        
    fig.colorbar(contour_plot)
    #plt.vlines(robust_pt, LB[1], UB[1], colors = 'black', linestyles = 'dashed')
    
    plt.scatter(xy_best[0], xy_best[1], marker = '*', color = 'white', s = 200, label = 'Nominal Solution')
    
    
    plt.xlim(LB[0], UB[0])
    plt.ylim(LB[1], UB[1])
    
    
    plt.xlabel('$z_{1}$', fontsize = 20)
    plt.ylabel('$z_{2}$', fontsize = 20)
    
    if sanity_check:
        def function_network(X: Tensor):
            return dropwave.evaluate(X=X).sum(dim = -1)
        
        # Sanity check:
        a = function_network(torch.tensor([[0.1, 0.1 ,0.5, torch.pi]]))
        b = fun(torch.tensor([[0.1 - 0.5 ,0.1]]))
        
        print(a - b)
