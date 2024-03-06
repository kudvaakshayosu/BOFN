#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:10:57 2024

@author: kudva.7
"""

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import standardize
import matplotlib.pyplot as plt
from botorch.models.transforms import Standardize
from botorch.acquisition import FixedFeatureAcquisitionFunction


# Define the function to optimize (for demonstration purposes)
def objective_function(X):
    return torch.sin(6 * X[:, 0]) + torch.cos(8 * X[:, 1]) + 0.1 * torch.randn(X.size(0))

# Define the bounds of the search space
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

# Generate initial training data
train_X = torch.rand(10,2)
train_Y = objective_function(train_X).unsqueeze(1)

# Construct the GP model
gp = SingleTaskGP(train_X, train_Y, outcome_transform = Standardize(m = 1))

# Fit the GP model
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# Define the acquisition function (Expected Improvement)
EI = ExpectedImprovement(gp, best_f=train_Y.min(), maximize=False)

# Optimize the acquisition function to find the next point to evaluate


columns = [1]

X = train_X.unsqueeze(1)

values = X[..., columns].squeeze(1)
qEI_FF = FixedFeatureAcquisitionFunction(EI, 2, columns, [torch.tensor(0.3996)])

#qei = qEI_FF(X[...,0].unsqueeze(1))



bounds2 = torch.tensor([[0.0], [1.0]])


candidate, acq_value = optimize_acqf(
    qEI_FF, bounds=bounds2, q=1, num_restarts=5, raw_samples=20)