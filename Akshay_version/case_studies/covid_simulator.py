import os
import sys
import torch
from torch import Tensor

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + '/group_testing/src')
sys.path.append('/' + script_dir.strip('/bofn_case_studies'))

from dynamic_protocol_design import simple_simulation



class CovidSimulator:
    def __init__(self, n_periods, seed=1):
        self.n_periods = n_periods
        self.seed =  seed
        self.input_dim = n_periods
        self.n_nodes = 3 * n_periods
        
    def evaluate(self, X):
        if X.dim() == 1:
            X_scaled = 199. * X[0:self.n_periods] + 1. 
            test_period = X[self.n_periods]
            prevalence = X[self.n_periods + 1]
            output = torch.zeros(torch.Size([self.n_nodes]))
            
            states, losses = simple_simulation(list(X_scaled), seed=self.seed, prevalence = float(prevalence), test_period = test_period)     
            states = torch.tensor(states)
            losses = torch.tensor(losses)
            for t in range(self.n_periods):
                output[3 * t] = states[t, 0]
                output[3 * t + 1] = states[t, 1]
                output[3 * t + 2] = losses[t]
        else:
            X_scaled = 199. * X[:,0:self.n_periods] + 1. 
            test_period = X[:,self.n_periods]
            prevalence = X[:,self.n_periods + 1]
            input_shape = X_scaled.shape
            output = torch.zeros(input_shape[:-1] + torch.Size([self.n_nodes]))
            for i in range(input_shape[0]):
                states, losses = simple_simulation(list(X_scaled[i, 0:self.n_periods]), seed=self.seed, prevalence = float(prevalence[i]), test_period= test_period[i])     
                states = torch.tensor(states)
                losses = torch.tensor(losses)
                for t in range(self.n_periods):
                    output[i, 3 * t] = states[t, 0]
                    output[i, 3 * t + 1] = states[t, 1]
                    output[i, 3 * t + 2] = losses[t]


        return output

if __name__ == '__main__':
    n_periods = 3
    problem = 'covid_' + str(n_periods)
    covid_simulator = CovidSimulator(n_periods=n_periods, seed=1)
    input_dim = covid_simulator.input_dim
    n_nodes = covid_simulator.n_nodes

    def function_network2(X: Tensor) -> Tensor:        
        # Testing period ranges between 12 and 15 days
        X[:,covid_simulator.n_periods] = 12 + X[:,covid_simulator.n_periods]*3        
        # Prevalence uncertainty ranges between 0.5% to 2%
        X[:,covid_simulator.n_periods + 1] = 0.005 + X[:,covid_simulator.n_periods + 1]*0.01 
        
        return covid_simulator.evaluate(X)

    # Underlying DAG
    parent_nodes = []
    for i in range(3):
        parent_nodes.append([])
    for t in range(1, n_periods):
        for i in range(3):
            parent_nodes.append([(t - 1) * 3, (t - 1) * 3 + 1])

    # Active input indices
    active_input_indices = []

    for t in range(n_periods):
        for i in range(3):
            active_input_indices.append([t, n_periods, n_periods + 1])

    verify_dag_structure = True
    # if verify_dag_structure:
    #     print(parent_nodes)
    #     print(active_input_indices)
    #     X = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.4], [0.1, 0.3, 0.3]])
    #     Y = function_network((X))
    #     print(Y)
    
    if verify_dag_structure:
        print(parent_nodes)
        print(active_input_indices)
        X = torch.tensor([[0.1, 0.2, 0.3, 0.33, 0.5], [0.1, 0.2, 0.4, 0.66, 1.], [0.1, 0.3, 0.3, 0.99, 0.]])
        Y = function_network2((X))
        print(Y)

    # Function that maps the network output to the objective value

    def network_to_objective_transform(Y):
        return -100 * torch.sum(Y[..., [3*t + 2 for t in range(n_periods)]], dim=-1)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    