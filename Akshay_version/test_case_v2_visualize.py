#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:42:25 2024

@author: kudva.7
"""
import torch
import matplotlib.pyplot as plt

####################
#fun = lambda x: 10/(1+ torch.log(x[:,0] + x[:,1]))

fun = lambda x: 100*(torch.sin(4.7/(1+ torch.log(x[:,0] + x[:,1]))) + 2*torch.cos(0.5*x[:,1]))

LB = torch.tensor([4,-1])
UB = torch.tensor([20,1])
####################

# fun = lambda x: -1*(torch.sin(x[:,0]*x[:,1]) + x[:,1].sqrt()*(x[:,0])**2 - 0.5*x[:,0])

# LB = torch.tensor([-1,2])
# UB = torch.tensor([2,4])

#######################################
    
    
    
xaxis = torch.arange(LB[0], UB[0], (UB[0]-LB[0])/500)
yaxis = torch.arange(LB[1], UB[1], (UB[1]-LB[1])/500)

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


####
fig, ax = plt.subplots(1, 1)
plt.set_cmap("jet")
contour_plot = ax.contourf(x2,y2,results2)
fig.colorbar(contour_plot)
plt.vlines(robust_pt, LB[1], UB[1], colors = 'black', linestyles = 'dashed')
plt.xlabel('z')
plt.ylabel('w')



