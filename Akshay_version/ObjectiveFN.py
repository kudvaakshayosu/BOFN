#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:38:38 2024
use a test case for simple network gp
@author: kudva.7
"""
import sys
sys.path
sys.path.append('/home/kudva.7/Desktop/PaulsonLab/BOFN/Akshay_version/case_studies')
from graph_utils import Graph
import torch
from torch import Tensor
from case_studies.robot_pushing import Robot_push
from case_studies.HeatX_simulator import HeatX
from case_studies.Cliff_simulation import Cliff
from case_studies.Rosenbrock_simulation import Rosenbrock
from case_studies.Sine_simulation import Sine
from case_studies.Polynomial_simulation import Polynomial
from case_studies.mnist_classification import NN_Classifier, mnist_tester
#from case_studies.group_testing.src.dynamic_protocol_design import simple_simulation
import copy
from botorch.utils.safe_math import smooth_amax, smooth_amin
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.double)


def function_network_examples(example, algorithm_name = 'BONSAI'):
    
        
    if example == 'synthetic_fun1_discrete':
             
        f1 = lambda x: torch.log(x[:,0] + x[:,1])
        f2 = lambda x: 10/(1+x)
    
        
        input_dim = 2
        
        
        def function_network(X: Tensor) -> Tensor:
            """
            Function Network: f1 --> f2 --> f3
            Parameters
            ----------
            X : X[0] is design variable -- Tensor
                X[1] is uncertain variable -- Tensor
        
            Returns
            -------
            Tensor
                Obtains a torch tensor
        
            """
            x_min = torch.tensor([4,-1])
            x_max = torch.tensor([20,1])
            
            X_scale = x_min + (x_max - x_min)*X
            
            #print(X_scale)
            try:
                f0_val = f1(X_scale)
            except:
                f0_val = f1(X_scale.unsqueeze(0))
            f1_val = f2(f0_val) 
    
            return torch.hstack([f0_val,f1_val])
        
        g = Graph(2)
        g.addEdge(0, 1)
        
        active_input_indices = [[0,1],[]]
        g.register_active_input_indices(active_input_indices=active_input_indices)
        
        uncertainty_input = [1]
        g.register_uncertainty_variables(uncertain_input_indices=uncertainty_input)
        
        # list of lists
        w1_set = [[0,0.1,0.2,0.7,1]] # Needs to be a list of list
        w_discrete_indices = [1]
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        nominal_w = torch.tensor([[0.7]])
    
    elif example == 'cliff':
        """
        High dimensional cliff function
        """
        
        simulator = Cliff()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes
        
        def function_network(X: Tensor) -> Tensor:  
            
            # Rescale things
            LB = torch.tensor([0.,0.,0.,0.,0.,-torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2])
            UB = torch.tensor([5.,5.,5.,5.,5.,torch.pi/2,torch.pi/2,torch.pi/2,torch.pi/2,torch.pi/2])
            
            X_scaled = LB + (UB - LB)*X
                
            return simulator.evaluate(X_scaled)
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes)    
        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[0,5],[1,6],[2,7],[3,8],[4,9]]
    
        
        ##############################################################
        g.register_active_input_indices(active_input_indices)
        
        uncertainty_input = [5,6,7,8,9]
        g.register_uncertainty_variables(uncertainty_input)
        
        # list of lists
        w1_set = [[0.,0.5,1.],[0.,0.5,1.],[0.,0.5,1.],[0.,0.5,1.],[0.,0.5,1.]] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        objective_function = lambda Y: torch.sum(Y[..., [t for t in range(n_nodes)]], dim=-1)
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.5, 0.5, 0.5, 0.5 , 0.5]])
        
    elif example == 'sine':
        """
        two dimensional sine function
        """
        
        simulator = Sine()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes
        
        def function_network(X: Tensor) -> Tensor:  
            
            # Rescale things
            LB = torch.tensor([-1.,-1.,-0.25,-0.25])
            UB = torch.tensor([1.,1.,0.25,0.25])
            
            X_scaled = LB + (UB - LB)*X
                
            return simulator.evaluate(X_scaled)
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes)  
        
        g.addEdge(0, 1)
        g.addEdge(0, 2)
        
        g.addEdge(3, 4)
        g.addEdge(3, 5)
        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[0,2],[],[],[1,3],[],[]]
    
        
        ##############################################################
        g.register_active_input_indices(active_input_indices)
        
        uncertainty_input = [2,3]
        g.register_uncertainty_variables(uncertainty_input)
        
        # list of lists
        #list_vals = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        list_vals = [0.,0.33,0.5,0.66,1.]
        w1_set = [list_vals,list_vals] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        objective_function = lambda Y: torch.sum(Y[..., [1,2,4,5]], dim=-1)
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.5, 0.5]])
        
    elif example == 'polynomial':
        """
        In this example, we have two uncertain variables, namely: 
            1) r: radius
            2) theta: angle
        """
        
       
        
        simulator = Polynomial()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes

        def function_network(X: Tensor) -> Tensor:   
            
            # Design variables
            X[...,0] = torch.tensor(-0.5) + torch.tensor(3.75)*X[...,0]
            X[...,1] = torch.tensor(-0.5) + torch.tensor(4.75)*X[...,1]
            
            # Uncertain variables scale
            X[...,3] = X[...,3]*2*torch.pi #theta
            X[...,2] = X[...,2]*0.5 # r
                
            return simulator.evaluate(X)
        
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes)    
        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[0,2,3],[1,2,3],[0,1,2,3]]

        
        ##############################################################
        g.register_active_input_indices(active_input_indices)
        
        uncertainty_input = [2,3]
        g.register_uncertainty_variables(uncertainty_input)
        
        # list of lists
        w1_set = [[0,0.2,0.4,0.6,1.],[0.0000, 0.100, 0.1250, 0.2000, 0.2500, 0.300, 0.3750, 0.4500, 0.5000, 0.5700, 0.6250, 0.700, 0.7500, 0.8750, 0.9500, 1.0000]] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        objective_function = lambda Y: torch.sum(Y[..., [t for t in range(n_nodes)]], dim=-1)
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.0, 0.0]])
        
    elif example == 'rosenbrock':
         """
         In this example, we have two uncertain variables, namely: 
             1) r: radius
             2) theta: angle
         """
         
        
         
         simulator = Rosenbrock()
         input_dim = simulator.input_dim
         n_nodes = simulator.n_nodes

         def function_network(X: Tensor) -> Tensor:   
             
             LB = torch.tensor([-1, 0, -0.1])
             UB = torch.tensor([2, 2, 0.1])
             
             X_scaled = LB + (UB - LB)*X
                 
             return simulator.evaluate(X_scaled)
         
         
         #############################################################
         # Define the graph for the problem
         g = Graph(n_nodes)  
         
         g.addEdge(0, 2)
         
         
         ###########################################################
         # Active input indices
         active_input_indices = [[0,2],[0,2],[1]]

         
         ##############################################################
         g.register_active_input_indices(active_input_indices)
         
         uncertainty_input = [1,2]
         g.register_uncertainty_variables(uncertainty_input)
         
         # list of lists
         w1_set = w1_set = [list(torch.linspace(0,1,20).detach().numpy()), list(torch.linspace(0,1,3).detach().numpy()) ] # Needs to be a list of list
         w_discrete_indices = uncertainty_input
         g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
         
         objective_function = lambda Y: -100*Y[...,2] - 1*Y[...,1]
         
         g.define_objective(objective_function)
         
         nominal_w = torch.tensor([[0.8, 0.5]])
        
    elif example == 'HeatX':
        """
        In this example, we have one second level variable, namely: 
            
        """
               
        
        simulator = HeatX()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes

        def function_network(X: Tensor) -> Tensor:   
            
            # Design variables
            rho = 5.
            LB = torch.tensor([620. - rho,388.- rho,583. - rho,313. - rho, 30.])
            UB = torch.tensor([620. + rho,388.+ rho,583. + rho,313. + rho, 150.])
            
            # Uncertain variables scale
            X_new =  LB + (UB - LB)*X
                
            return simulator.evaluate(X_new)
        
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes)    
        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[1,4],[0,1,2,4],[0,1,2,4],[0,1,2,3,4],[0,1,2,3,4]]

        
        ##############################################################
        g.register_active_input_indices(active_input_indices)
        
        uncertainty_input = [4]
        g.register_uncertainty_variables(uncertainty_input)
        
        # list of lists
        w1_set = [list(torch.arange(0,1,0.0001).detach().numpy())] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        #objective_function = lambda Y: smooth_amax(Y, dim=-1)
        
        objective_function = lambda Y: torch.max(Y, dim=-1).values
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.5]])
        
        #g.figure()
        
    elif example == 'robot':
        """
        In this example, we have one uncertain variable, namely: 
            1) object friction
     """       
        simulator = Robot_push()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes

        def function_network(X: Tensor):
            if algorithm_name == 'ARBO':
                X = X[...,[0,1,2]]
            
            x_min = torch.tensor([-5.,-5.,1.])
            x_max = torch.tensor([5.,5.,70.])  
            val = 0.
            
            if X.size()[0] == 1 or X.dim() == 1:  
                if X.dim() == 1:
                    X = X.unsqueeze(0)
                X_scaled = x_min + (x_max - x_min)*X  
                val = simulator.evaluate(X=X_scaled)
            else:
                X_scaled = x_min + (x_max - x_min)*X 
                Y = torch.empty(X.size()[0], simulator.n_nodes)
                #print(X_scaled)
                i = 0
                for x in X_scaled:  
                    Y[i] = simulator.evaluate(X=x.unsqueeze(0))
                    i += 1            
                val = Y        
            
            return val
        
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes) 
        
        g.addEdge(0, 2)
        g.addEdge(0, 3)
        
        g.addEdge(1, 2)
        g.addEdge(1, 3)
        
        g.addEdge(2, 5)
        g.addEdge(2, 4)
        
        g.addEdge(3, 4)
        g.addEdge(3, 5)        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[0,1,2],[0,1,2],[2],[2],[2],[2]]
        g.register_active_input_indices(active_input_indices)        
       
        
        ##################################################################
        # Generate a normal distribution
        torch.manual_seed(100)
        mean = torch.tensor([0., 0.])  # Mean of the distribution
        covariance = torch.tensor([[0.6, - 0.4], [ - 0.4, 0.6]])  # Covariance matrix
          

        # Create a multivariate normal distribution
        mv_normal = torch.distributions.MultivariateNormal(mean, covariance)
        # Create a multivariate normal distribution
        #mv_normal2 = torch.distributions.MultivariateNormal(mean2, covariance2)

        # Generate random samples from the normal distribution
        a = torch.tensor([1.5,1.])
        b = torch.tensor([2.5,5.])
        
        box_mean = (a + b)/2
        samples = mv_normal.sample((20,))
        samples2 = (a - b)*torch.rand(300,2) + b # Generate samples from rectangle
        
        targets = torch.vstack((samples,samples2))            
            

        # # Plot the generated samples
        # plt.figure(figsize=(8, 6))
        # plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        # plt.scatter(samples2[:, 0], samples2[:, 1], alpha=0.5, color = 'red')

        # plt.title('Two-Dimensional Normal Distribution')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid(True)
        # plt.show()
        
        
        # This is for the norm with respect to each point from the uncertain set
        if algorithm_name == 'BONSAI':
            objective_function = lambda Y: smooth_amin(-1*torch.sqrt(torch.sum((Y[..., [4,5]].unsqueeze(0) - targets.unsqueeze(1))**2, -1)), dim = 0)
            g.define_objective(objective_function)
        #objective_function = lambda Y: -1*torch.sqrt(torch.sum((Y[..., [4,5]] - targets.mean(dim = 0))**2, -1))
        elif algorithm_name == 'BOFN' or algorithm_name == 'VBO':
            objective_function = lambda Y: -1*torch.sqrt(torch.sum((Y[..., [4,5]] - targets.mean(dim = 0))**2, -1))
            #objective_function = lambda Y: -1*torch.sqrt(torch.sum((Y[..., [4,5]] - (mean + box_mean)/2)**2, -1))
            g.define_objective(objective_function)
        
        # TODO: Define an objective function and update g.w_conbinations
        elif algorithm_name == 'ARBO':
            t_min = targets.min(0).values
            t_max = targets.max(0).values
            
            t_vals = (targets - t_min)/(t_max - t_min)
            
            g.w_sets = [list(t_vals[:,0].detach().numpy()),list(t_vals[:,1].detach().numpy())]
            g.w_combinations = t_vals
            
            
            objective_function = lambda X, Y : -1*torch.sqrt(torch.sum((Y[..., [4,5]] - (t_min + X[...,[3,4]]*(t_max - t_min)))**2, -1))
            g.define_objective(objective_function, type_obj= 'kkk')
            
        elif algorithm_name == 'Recommender':
            objective_function = lambda Y: torch.min(-1*torch.sqrt(torch.sum((Y[..., [4,5]].unsqueeze(0) - targets.unsqueeze(1))**2, -1)), dim = 0)
            g.define_objective(objective_function)
            
       
        nominal_w = None
        #g.figure()   
        
    elif example == 'classifier':
        
        simulator = NN_Classifier()
        input_dim = simulator.input_dim
        n_nodes = simulator.n_nodes
        
        if algorithm_name == 'BOFN' or algorithm_name == 'VBO':
            adversary_attack = False
        else:
            adversary_attack = True
            
        
        #############################################################
        # Define the graph for the problem
        g = Graph(n_nodes) 
        
        
        ###########################################################
        # Active input indices
        active_input_indices = [[0,1,2,3],[0,1,2,4]]
        g.register_active_input_indices(active_input_indices) 
        
        uncertainty_input = [3,4]
        g.register_uncertainty_variables(uncertainty_input)
       
        
        ##################################################################
        
        
        def function_network(X: Tensor, test_mode = False):
            
            # x_min = torch.tensor([1e-3,10.,0., 0.,0.])
            # x_max = torch.tensor([0.5,50.,5.,1.,1.])  
            
            x_min = torch.tensor([1e-4,10.,0., 0.,0.])
            x_max = torch.tensor([0.1,30.,10.,1.,1.])  
            
            
            if X.size()[0] == 1 or X.dim() == 1:  
                if X.dim() == 1:
                    X = X.unsqueeze(0)
                X_scaled = x_min + (x_max - x_min)*X  
                val = simulator.evaluate(X=X_scaled, adversary_attack= adversary_attack)
            else:
                X_scaled = x_min + (x_max - x_min)*X 
                Y = torch.empty(X.size()[0], simulator.n_nodes)
                #print(X_scaled)
                i = 0
                if test_mode:
                    x = X_scaled[0]
                    Y = mnist_tester(X = x.unsqueeze(0), adversary_attack= adversary_attack)
                    val = Y
                    
                else:              
                    for x in X_scaled:  
                        Y[i] = simulator.evaluate(X=x.unsqueeze(0),adversary_attack= adversary_attack)
                        i += 1            
                    val = Y                    
            return val
        
        #################################################################
        
        w1_set = [[0.,0.33,0.66,1.],[0.,0.33,0.66,1.]] # Needs to be a list of list
        w_discrete_indices = uncertainty_input
        g.register_discrete_uncertain_values(vals = w1_set, indices = w_discrete_indices)
        
        objective_function = lambda Y: torch.mean(Y, dim=-1)
        
        g.define_objective(objective_function)
        
        nominal_w = torch.tensor([[0.66,0.33]])   # This value does not really matter in this case study, we are setting no adversarial attack nominal case   
        
    
    else:
        print(' Enter a valid test function')
        
    return  function_network, g, nominal_w


    
if __name__ == '__main__':
    
    example_list = ['synthetic_fun1_discrete', 'covid_testing', 'polynomial', 'robot']
    example = example_list[3]
    
    function_network, g, _ = function_network_examples(example)
    
    test_rest = True
    
    if test_rest: 
        
        uncertainty_input = g.uncertain_input_indices
        design_input = g.design_input_indices    
        
        input_index_info = [design_input, uncertainty_input]
        
        nz = g.nz
        nw = g.nw
        input_dim = g.nx   
        
        #g.figure()
        
        torch.manual_seed(1000)
        x_test =  torch.rand(1,input_dim)  
        
        # Test if everything is okay
        y_true = function_network(x_test)   
        
        
        
        
        # Start the modeling procedure
        Ninit = 5
        n_outs = g.n_nodes
        
        
        
        
        x_init = torch.rand(Ninit, input_dim)
        y_init = torch.zeros(Ninit,n_outs)
        
        for i in range(Ninit):
            y_init[i] = function_network(x_init[i])
            
            
    
   
