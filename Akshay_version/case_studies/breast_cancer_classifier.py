#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:37:12 2024

@author: kudva.7
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import pandas as pd

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_min = np.min(X,0)
X_max = np.max(X,0)

df_features = pd.DataFrame(data.data, columns=data.feature_names)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42, C = 0.5)

# Train the SVM classifier
t1 = time.time()
svm_classifier.fit(X_train, y_train)
t2 = time.time()
print(t2 - t1)

# Perturb
X_test +=  0.05*(2*np.random.rand(1,30) - 1)*(X_max - X_min)

# Predict the labels for the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


