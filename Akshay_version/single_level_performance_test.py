#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:39:13 2024

@author: kudva.7
"""

import torch
from botorch.settings import debug
from torch import Tensor
import time
from gp_network_utils import GaussianProcessNetwork
from torch import Tensor
from typing import Optional
from botorch.optim.initializers import gen_batch_initial_conditions
from utils import generate_initial_data
from single_level_Objective import function_network_examples
from single_level_algorithms import BOFN, BayesOpt, BONS


examples = ['dropwave']
acq_type = ['qEI','qlogEI','qUCB', 'UCB']
acq_fun = acq_type[2]

example = examples[0]


# Generate initial data

function_network, g = function_network_examples(example = example)

x_init, y_init = generate_initial_data( g = g, function_network = function_network, Ninit = 10, seed = 2000)

#val = BOFN( x_init, y_init, g, objective = function_network, T = 10, acq_type = acq_fun)
#val = BayesOpt( x_init, y_init, g, objective = function_network, T = 10, acq_type = acq_fun)
val = BONS( x_init, y_init, g, objective = function_network, T = 10)













