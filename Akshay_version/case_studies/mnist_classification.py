#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:43:50 2024

@author: kudva.7
"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
import numpy as np

import torch

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


class NN_Classifier:
    def __init__(self):
        self.n_nodes = 2
        self.input_dim = 5

    def evaluate(self, X, adversary_attack = True):        
        output = mnist_classifier(X, adversary_attack)        
        return output



def mnist_classifier(X, adversary_attack = True):
    
    """
    MNIST classifier:
    
    X[0] = learning_rate
    X[1] = hidden layer size
    X[2] = alpha
    
    X[3] = adversarial attack with gaussian noise       
    
    """
    
    # Clean up the data
    
    X = X.detach().numpy()[0]
    
    learning_rate = float(X[0])    
    hidden_layer_sizes = (int(round(X[1])),int(round(X[1])),)
    #hidden_layer_sizes = (50,50,)
    alpha = float(X[2])
    
    adversary_case1 = float(X[3])
    adversary_case2 = float(X[4])
    
    
    mylist = [0, 0.33, 0.66, 1]
    
    adversary_case1 = min(mylist, key=lambda x:abs(x-adversary_case1))
    adversary_case2 = min(mylist, key=lambda x:abs(x-adversary_case2))
    
    adversary_case1 = mylist.index(adversary_case1)
    adversary_case2 = mylist.index(adversary_case2)
    
    # Generate the dataset values and train based on values
    
    mnist = load_digits()
    
    X_data, y_data = mnist.data, mnist.target
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    
    
    # Define hyperparameters
    # learning_rate = 0.001  # Example value, specify as desired
    # hidden_layer_sizes = (300,)  # Example value, specify as desired
    # alpha = 0.0001  # Example value, specify as desired
    
    # Define and train the classifier
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        alpha=alpha,
        random_state=42,
        max_iter = 10000,
    )
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier on the test set
    test_shape = np.shape(X_test)
    
    # Predict on training set
    y_train_pred = clf.predict(X_train)
    
    # Evaluate training performance
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_classification_report = classification_report(y_train, y_train_pred, output_dict= True)
    
    
    # Switch to adversarial case
    np.random.seed(seed = 42)
    adversary1 = np.random.rand(test_shape[0], int(test_shape[1]/4))*5
    np.random.seed(seed = 42)
    adversary2 = np.random.rand(test_shape[0], int(test_shape[1]/4))*5
    
    # Two adversaries
    total_adversary1 = np.zeros(test_shape)
    total_adversary2 = np.zeros(test_shape)
   
    total_adversary1[:, adversary_case1*int(test_shape[1]/4):(adversary_case1 + 1)*int(test_shape[1]/4)] = adversary1
    
    if adversary_case2 == 0:
        total_adversary2[:,:8] = adversary2[:,:8]
        total_adversary2[:,16:24] = adversary2[:,8:]
        
    elif adversary_case2 == 1:
        total_adversary2[:,8:16] = adversary2[:,:8]
        total_adversary2[:,24:32] = adversary2[:,8:]
        
    elif adversary_case2 == 2:
        total_adversary2[:,32:40] = adversary2[:,:8]
        total_adversary2[:,48:56] = adversary2[:,8:]
    else:
        total_adversary2[:,40:48] = adversary2[:,:8]
        total_adversary2[:,56:64] = adversary2[:,8:]
    
    
   #total_adversary2[:, adversary_case2*int(test_shape[1]/4):(adversary_case2 + 1)*int(test_shape[1]/4)] = adversary2
    
     
    if adversary_attack:
        X_test1 = X_test + total_adversary1
        X_test2 = X_test + total_adversary2
    else:
        X_test1 = X_test
        X_test2 = X_test 
        
    
    test_accuracy1 = clf.score(X_test1, y_test)
    test_accuracy2 = clf.score(X_test2, y_test)
    #print("Test Accuracy:", test_accuracy)
    
    
    
    #val3 = train_classification_report['weighted avg']['f1-score']
    
    return torch.tensor([test_accuracy1, test_accuracy2])




def mnist_tester(X, adversary_attack = True):
    
    """
    MNIST classifier:
    
    X[0] = learning_rate
    X[1] = hidden layer size
    X[2] = alpha
    
    X[3] = adversarial attack with gaussian noise       
    
    """
    
    # Clean up the data
    
    X = X.detach().numpy()[0]
    
    learning_rate = float(X[0])    
    hidden_layer_sizes = (int(round(X[1])),int(round(X[1])),)
    #hidden_layer_sizes = (50,50,)
    alpha = float(X[2])
    
    adversary_case1 = float(X[3])
    adversary_case2 = float(X[4])
    
    
    mylist = [0, 0.33, 0.66, 1]
    
    adversary_case1 = min(mylist, key=lambda x:abs(x-adversary_case1))
    adversary_case2 = min(mylist, key=lambda x:abs(x-adversary_case2))
    
    adversary_case1 = mylist.index(adversary_case1)
    adversary_case2 = mylist.index(adversary_case2)
    
    # Generate the dataset values and train based on values
    
    mnist = load_digits()
    
    X_data, y_data = mnist.data, mnist.target
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    
    
    # Define hyperparameters
    # learning_rate = 0.001  # Example value, specify as desired
    # hidden_layer_sizes = (300,)  # Example value, specify as desired
    # alpha = 0.0001  # Example value, specify as desired
    
    # Define and train the classifier
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        alpha=alpha,
        random_state=42,
        max_iter = 10000,
    )
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier on the test set
    test_shape = np.shape(X_test)
    
    # Predict on training set
    y_train_pred = clf.predict(X_train)
    
    # Evaluate training performance
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_classification_report = classification_report(y_train, y_train_pred, output_dict= True)
    
    
    # Switch to adversarial case
    np.random.seed(seed = 42)
    adversary1 = np.random.rand(test_shape[0], int(test_shape[1]/4))*5
    np.random.seed(seed = 42)
    adversary2 = np.random.rand(test_shape[0], int(test_shape[1]/4))*5
    
    Y = torch.empty(16,2)
    i = 0
    for adversary_case1 in range(4):
        for adversary_case2 in range(4):
            # Two adversaries
            total_adversary1 = np.zeros(test_shape)
            total_adversary2 = np.zeros(test_shape)
           
            total_adversary1[:, adversary_case1*int(test_shape[1]/4):(adversary_case1 + 1)*int(test_shape[1]/4)] = adversary1
            
            if adversary_case2 == 0:
                total_adversary2[:,:8] = adversary2[:,:8]
                total_adversary2[:,16:24] = adversary2[:,8:]
                
            elif adversary_case2 == 1:
                total_adversary2[:,8:16] = adversary2[:,:8]
                total_adversary2[:,24:32] = adversary2[:,8:]
                
            elif adversary_case2 == 2:
                total_adversary2[:,32:40] = adversary2[:,:8]
                total_adversary2[:,48:56] = adversary2[:,8:]
            else:
                total_adversary2[:,40:48] = adversary2[:,:8]
                total_adversary2[:,56:64] = adversary2[:,8:]  
     

            X_test1 = X_test + total_adversary1
            X_test2 = X_test + total_adversary2           
        
            test_accuracy1 = clf.score(X_test1, y_test)
            test_accuracy2 = clf.score(X_test2, y_test)
            
           
            Y[i] = torch.tensor([test_accuracy1, test_accuracy2])
            i += 1
            
            
    #print("Test Accuracy:", test_accuracy)
    
    
    
    #val3 = train_classification_report['weighted avg']['f1-score']
    
    return Y










if __name__ == '__main__':
    a = 0
    
    
    digits = load_digits()
    
    _, axes = plt.subplots(nrows= 5, ncols= 4, figsize=(10, 3))   
    
    image = digits.images
    
    
    
    for i in range(5):
        for j in range(4):
            ax = axes[i,j]
            
            
            total_adversary = np.zeros((8,8))
            
            if i > 0:
                adversary = 20*np.random.rand(2, 8)
                total_adversary[2*i - 2:2*i,:] = adversary  
            else:
                ax.set_title("Training: %i" %j)
            
            img = image[j] + total_adversary 
            ax.set_axis_off()
            ax.imshow(img, cmap=plt.cm.gray_r, interpolation="lanczos")
            
    

    
    
    # k = torch.tensor([[0.01, 10, 0.33 , 1]])
    
    # b = mnist_classifier(k)
