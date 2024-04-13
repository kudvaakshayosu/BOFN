#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:27:44 2024

@author: kudva.7
"""

import torch


class HeatX:
    def __init__(self):
        self.n_nodes = 5
        self.input_dim = 4
        
    def evaluate(self, X):
        T1 = X[...,0]
        T3 = X[...,1]
        T5 = X[...,2]
        T8 = X[...,3]
        Qc = X[...,4]
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))

        output[...,0] = -0.67*Qc + T3 - 350
        output[...,1] = -T5 - 0.75*T1 + 0.5*Qc - T3 + 1388.5
        output[...,2] = -T5 - 1.5*T1 + Qc -2*T3 + 2044
        output[...,3] = -T5 - 1.5*T1 + Qc - 2*T3 - 2*T8 + 2830    
        output[...,4] = T5 + 1.5*T1 - Qc + 2*T3 + 3*T8 - 3153
        
        return output

if __name__ == '__main__':    
     a = HeatX()
     nominal = torch.tensor([620.,388.,583.,313.])
     LB = torch.tensor([620. - 2.,388.- 2.,583. - 2.,313. - 2., 3.])
     UB = torch.tensor([620. + 2.,388.+ 2.,583. + 2.,313. + 2., 150.])
     
     rand_init = LB + (UB - LB)*torch.rand(20,5)
     
     print(a.evaluate(rand_init))