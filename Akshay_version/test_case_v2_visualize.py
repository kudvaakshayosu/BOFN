#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:42:25 2024

@author: kudva.7
"""
import torch
import matplotlib.pyplot as plt


fun = lambda x: 10/(1+ torch.log(x[:,0] + x[:,1]))

LB = torch.tensor([4,-1])
UB = torch.tensor([20,1])

xaxis = torch.arange(LB[0], UB[0], (UB[0]-LB[0])/200)
yaxis = torch.arange(LB[1], UB[1], (UB[1]-LB[1])/200)

# create a mesh from the axis
x2, y2 = torch.meshgrid(xaxis, yaxis)

# reshape x and y to match the input shape of fun
xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)


results = fun(xy)

results2 = results.reshape(x2.size())

####
fig, ax = plt.subplots(1, 1)
plt.set_cmap("jet")
contour_plot = ax.contourf(x2,y2,results2)
fig.colorbar(contour_plot)
plt.xlabel('z')
plt.ylabel('w')



