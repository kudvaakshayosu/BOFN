#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:11:46 2024

@author: kudva.7
"""


import networkx as nx
import random
from graph_utils import Graph
import matplotlib.pyplot as plt
from case_studies.generate_TS import generateTS
import torch
from torch import Tensor

def generate_random_dag(n, m, seed = None):    
    if seed:
        random.seed(seed)
    
    if m < n - 1:
        raise ValueError("Invalid number of edges. A DAG must have at least n-1 edges.")
    
    G = Graph(n)
    
    # Ensure the graph is connected by adding edges to form a tree
    for i in range(1, n):
        parent = random.choice(range(i))
        G.addEdge(parent, i)
    
    # Add remaining edges randomly, but ensuring each node (except root) has exactly one incoming edge
    remaining_edges = m - (n - 1)
    while remaining_edges > 0:
        u = random.randint(1, n - 1)  # Exclude root node (0) to ensure only one input per node
        v = random.randint(0, u - 1)  # Incoming edge comes from nodes with lower index
        if not G.has_edge(v, u):
            G.addEdge(v, u)
            remaining_edges -= 1    
    return G


class random_graph_objective_function:   
    """
    Parameters
    ----------
    n_bounds: list 
        Contains the lower bound at index 0 and upper bound at index 1 for the number
        of vertices  
   
    """
    def __init__(self, n_bounds, input_dims = 2):
        self.num_nodes = random.randint(n_bounds[0], n_bounds[1])
        self.num_edges = random.randint(self.num_nodes - 1, self.num_nodes + 5) # Only set to keep the graph sparse    
        self.input_dims = input_dims  
     
    def generate_objective(self, seed = None):
        self.g = generate_random_dag(self.num_nodes, self.num_edges, seed = seed)
        
        active_input_indices = [[] for i in range(self.num_nodes)]        
        input_indices = [i for i in range(self.input_dims)]
        
        active_input_indices[0] = input_indices
        
        #print(active_input_indices)
        # Register the active input indices
        self.g.register_active_input_indices(active_input_indices)
        
        # We add more edges to have only the final node to be child node
        # Here we look for nodes other than the final node number that do not have children
        # In a way we are having the childless node "adopt" the final node :P
        graph = self.g.graph
        
        list_of_parents = [] # create a list of parent nodes
        
        for parents in graph:
            if len(graph[parents]) != 0:
                list_of_parents.append(parents)
        
        # Lets adopt the final node value
        
        for i in range(self.num_nodes - 1):
            if i in list_of_parents:
                continue
            else:
                self.g.addEdge(i,self.g.n_nodes - 1)        
        self.g.figure()    
        
        # Assign thompson samples for each node:
        self.thompson_samples = [[] for i in range(self.g.n_nodes)]
        self.thompson_samples[0] =  generateTS(self.input_dims)
        
        for i in range(1,self.num_nodes):
            self.thompson_samples[i] = generateTS(len(self.g.parent_nodes))              
        print('Thompson samples for each node generated')
        
        
    # TODO: evaluate thompson samples with a sweep through the graph  
    def evaluate(self, X):
        
        # Get a tensor of output values
        Y = torch.empty(self.num_nodes)     
        
        input_vals = [torch.empty(len(self.g.parent_nodes[i])) for i in range(self.num_nodes)]
        input_vals_counter = [0 for i in range(self.num_nodes)] # Keeps track of all the values
        
        # Get the values for 
        input_vals[0] = X    
        
        #for i in range(self.n_nodes):      
        return 0
    




if __name__ == "__main__":
    # Example usage
    num_nodes_bounds = [5,10]
    random_graph =  random_graph_objective_function(num_nodes_bounds)   
    random_graph.generate_objective(seed = 2000)
    random_graph.evaluate(torch.tensor([0.2,0.5]))