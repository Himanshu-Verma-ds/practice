import pandas as pd 
import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import optuna 
from functools import partial

class MyModel(nn.Module):
    def __init__(self, input_features, hidden_neurons, output_features):
        super().__init__()
        self.input_features= input_features
        self.hidden_neurons= hidden_neurons
        self.output_features= output_features
        self.linear1= nn.Linear(input_features, hidden_neurons)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer= nn.Linear(hidden_neurons, output_features)
        
    def forward(self, x):
        x= self.linear1(x)
        x= self.relu(x)
        x= self.linear2(x)
        x= self.relu(x)
        x= self.output_layer(x)
        return x
    

class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features= torch.tensor(features, dtype= torch.float32)
        self.target= torch.tensor(target, dtype= torch.float32)
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
    
    
    
def train_and_validate(model, train_dataloader, val_dataloader, epochs, optimizer, loss_fn, device):
    
    model= model.train()
    model= model.to(device)
    for epoch in range(epochs):
        total_epoch_loss= 0
        
        for batch_features, batch_target in train_dataloader:
            batch_features, batch_target= batch_features.to(device), batch_target.to(device)
            
            y_pred= model(batch_features)
            loss= loss_fn(y_pred, batch_target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
        
        avg_train_loss= total_epoch_loss/len(train_dataloader)
        print(f'Epoch: {epoch+1}, Avg. loss per epoch: {avg_train_loss}')
    
    # Eval mode
    model.eval()
    val_loss=0
    with torch.no_grad():
        for batch_features, batch_target in val_dataloader:
            batch_features, batch_target= batch_features.to(device), batch_target.to(device)
            preds= model(batch_features)
            loss= loss_fn(preds, batch_target.view(-1))
            val_loss += loss.item()
    
    avg_val_loss= val_loss/len(val_dataloader)
    print('Average validation loss: ', avg_train_loss)
    return avg_val_loss
        
    

def evaluate_on_test(model, test_dataloader, device):
    
    model= model.to(device)
    loss_fn= nn.MSELoss()
    
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for batch_features, batch_target in test_dataloader:
            batch_features, batch_target= batch_features.to(device), batch_target.to(device)
            preds= model(batch_features)
            loss= loss_fn(preds, batch_target.view(-1))
            test_loss += loss.item()
    
    avg_test_loss= test_loss/len(test_dataloader)
    return avg_test_loss



test_dataloader_global = None
X_test_global= None 

def objective(trial, device):
    global test_dataloader_global
    global X_test_global
    
    hidden_neurons= trial.suggest_int('hidden_neurons', 16, 256, step= 16)
    lr= trial.suggest_float('lr', 1e-9, 1e-2, log= True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # train test split 
    X, y= make_regression(n_samples= 10000, n_features=5, noise= 0.1, random_state= 42)
    
    
    
    # Creating synthetic dataset
    n_samples= 10000
    n_features= 10
    noise_factor= 0.001
    
    X= torch.randn(n_samples, n_features)
    true_weights= torch.tensor([3.0, -2.5, 0.0, 0.0, 5.0, 0.0, 1.2, 0.0, 0.0, -4.0]) 
    
    noise= torch.randn(n_samples)*noise_factor
    y= torch.matmul(X, )
    
    df= pd.DataFrame(X, columns= [f'feature_{i}' for i in range(X.shape[1])])
    df['target']= y
    
    X_tmp, X_test, y_tmp, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)
    X_train, X_val, y_train, y_val= train_test_split(X_tmp, y_tmp, test_size= 0.2, random_state= 42)
    
    train_dataloader= DataLoader(CustomDataset(X_train, y_train), batch_size= batch_size, shuffle= True)
    val_dataloader= DataLoader(CustomDataset(X_val, y_val), batch_size= batch_size, shuffle= False)
    test_dataloader= DataLoader(CustomDataset(X_test, y_test))
    
    
    if test_dataloader_global is None:
        test_dataloader_global= test_dataloader
    
    model= MyModel(input_features= X_train.shape[1],
                   hidden_neurons= hidden_neurons,
                   output_features= 1)
    
    model= model.to(device)
    
    if X_test_global is None:
        X_test_global= X_test 
    
    
    epochs= 50
    loss_fn= nn.MSELoss()
    optimizer= torch.optim.Adam(model.parameters(), lr= lr)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_loss= train_and_validate(model= model,
                                 train_dataloader= train_dataloader,
                                 val_dataloader= val_dataloader,
                                 epochs= epochs,
                                 optimizer= optimizer,
                                 loss_fn= loss_fn,
                                 device= device)
    
    return val_loss
    


if __name__ == '__main__':
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    study= optuna.create_study(direction= 'minimize')
    study.optimize(partial(objective, device= device), n_trials= 30)
    
    print("\nBest Trial:")
    print(f"Validation Loss: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    # Predicting with the best parameters model

    model= MyModel(input_features= X_test_global.shape[1],
                   hidden_neurons= study.best_params['hidden_neurons'],
                   output_features= 1)
    
    avg_test_loss= evaluate_on_test(model= model, 
                                    test_dataloader= test_dataloader_global, 
                                    device= device)
    
    print('Test loss using the best model: ', avg_test_loss)