#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:04:38 2024

@author: kudva.7
"""
from push_world import *
import numpy as np
import torch
from torch import Tensor

# This is from the paper Max-value Entropy Search for Efficient Bayesian Optimization (2018) by Wang and Jegelka

class Robot_push:
    def __init__(self):
        self.n_nodes = 6
        self.input_dim = 3

    def evaluate(self, X):          
        return robot_simulate(X)



def robot_simulate(X):
    """
    Parameters
    ----------
    X : Tensor
        X[0]: rx (object position x axis)
        X[1]: ry (object position y axis)
        X[2]: simulation steps 
        X[3]: object friction

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    X = X.detach().numpy()[0]
    N = 1
    np.random.seed(1000)   
    
    #print(X)
    
    rx, ry = float(X[0]), float(X[1])
    simu_steps = int(round(float(X[2])))*10
    
    #gXY = 10 * np.random.rand(N, 2) - 5
    
    ofriction = 0.01
    odensity = 0.05
    
    #print(gXY)
    
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    #world2 = b2WorldInterface(False)
    oshape, osize, bfriction, hand_shape1, hand_size1 = 'circle', 1, 0.01,'rectangle',(0.3,1)  
      
    # Circular pusher
    thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))  
    init_angle = np.arctan(ry/rx)
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape1, hand_size1)
    ret1 = simu_push(world, thing, robot, base, simu_steps)
    #print(ret1)
    
    # Rectangular pusher
    # thing2,base2 = make_thing(500, 500, world2, oshape, osize, ofriction, odensity, bfriction, (0,0))
    # robot2 = end_effector(world2, (rx,ry), base2, init_angle, hand_shape2, hand_size2)
    # ret2 = simu_push(world2, thing2, robot2, base2, simu_steps)
    #print(ret2)
    

    return ret1


if __name__ == '__main__':
    
    robot = Robot_push() 
    
    def function_network(X: Tensor):
        x_min = torch.tensor([-5.,-5.,1.])
        x_max = torch.tensor([-2.,-3.,30.])  
        val = 0.
        
         
        X_scaled = x_min + (x_max - x_min)*X  
        val = robot.evaluate(X=X_scaled)  
        
        return val
    
    k = torch.tensor([[1.,1.,1]])
    b = function_network(k)

    # x_rand = np.array([5.,-2.,20])
    
    # k = robot_simulate(x_rand)
    # print(k)
    
    
    
    