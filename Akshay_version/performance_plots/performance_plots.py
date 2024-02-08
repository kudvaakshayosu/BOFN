#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:30:00 2024

@author: kudva.7
"""

import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('BONSAI_non_concave_twoD_extracted.pickle', 'rb') as handle:
    BONSAI_data = pickle.load(handle)
    
    
with open('ARBO_non_concave_twoD_extracted.pickle', 'rb') as handle:
    ARBO_data = pickle.load(handle)
 
    
data = BONSAI_data
data2 = ARBO_data
# For the non-concave function
soln = torch.tensor(275.3721)

plot_time = False
plot_simple_regret = True

if plot_time:

    # Convert the Torch tensor to a Pandas DataFrame
    df = pd.DataFrame(data['time'].t().numpy())
    
    # Set the seaborn style with grid
    sns.set_style("whitegrid")
    
    # Melt the DataFrame to have T on x-axis and values on y-axis
    df_melted = df.melt(var_name='Iteration,t', value_name='Time')
    
    # Plotting the box plot using Seaborn
    sns.boxplot(x='Iteration,t', y='Time', data=df_melted)
    
    plt.title('Time per iteration - ARBO')


tensor_data1 = data['F_min_W']
tensor_data2 = data2['F_min_W']

# Compute median, worst case (minimum), and best case (maximum) values
median_values = (soln -tensor_data1 )[5:,:].median(dim = 1).values

# Compute median, worst case (minimum), and best case (maximum) values
median_values2 = (soln -tensor_data2 )[5:,:].median(dim = 1).values

iteration_list = [i for i in range(10)]

plt.plot(iteration_list, median_values, color = 'green', label = 'BONSAI')
plt.plot(iteration_list, median_values2, color = 'red', label = 'ARBO')
plt.xlabel('Iteration, t')
plt.ylabel('Instantaneous robust regret over median')
plt.legend()




