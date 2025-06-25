from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np 
import os  
import pandas as pd
import torch 
import torch.nn as nn 


class Model(nn.Module):
    def __init__(self, input_features, hidden_neurons, output_features):
        super().__init__() 
        self.input_features= input_features
        self.hidden_neurons= hidden_neurons
        self.output_features= output_features
        
        self.linear1= nn.Linear(input_features, hidden_neurons)
        self.batch_norm= nn.BatchNorm1d(hidden_neurons)
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(p= 0.4)
        self.linear2= nn.Linear(hidden_neurons, output_features)
    
    def forward(self, x):
        x= self.linear1(x)
        x= self.batch_norm(x)
        x= self.relu(x)
        x= self.dropout(x)
        x= self.linear2(x)
        return x
            
            
X,y= make_regression(n_samples=5000, n_features= 10, noise=15, random_state= 42)
df= pd.DataFrame(X, columns= [f'feature_{i}' for i in range(X.shape[1])])
df['target']= y

print(df)


