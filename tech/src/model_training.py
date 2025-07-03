from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np 
import os  
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_features, hidden_neurons, output_features):
        super().__init__() 
        self.input_features= input_features
        self.hidden_neurons= hidden_neurons
        self.output_features= output_features
        
        self.linear1= nn.Linear(input_features, hidden_neurons)
        self.batch_norm= nn.BatchNorm1d(hidden_neurons)
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(p= 0.1)
        self.linear2= nn.Linear(hidden_neurons, hidden_neurons)
        self.linear3= nn.Linear(hidden_neurons, output_features)
    
    def forward(self, x):
        x= self.linear1(x)
        x= self.batch_norm(x)
        x= self.relu(x)
        x= self.dropout(x)
        x= self.linear2(x)
        x= self.batch_norm(x)
        x= self.relu(x)
        x= self.dropout(x)
        x= self.linear3(x)
        return x
            
            
# X,y= make_regression(n_samples=5000, n_features= 10, noise= 0.001, random_state= 42)

# Making the dataset
n_samples = 5000
n_features = 10
noise_factor= 0.001

X = torch.randn(n_samples, n_features)
true_weights = torch.tensor([3.0, -2.5, 0.0, 0.0, 5.0, 0.0, 1.2, 0.0, 0.0, -4.0]) 
noise = torch.randn(n_samples) * noise_factor

y = torch.matmul(X, true_weights) + noise
y = y.view(-1)    

df= pd.DataFrame(X, columns= [f'feature_{i}' for i in range(X.shape[1])])
df['target']= y
print(df.head(10))


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.shape)

device= torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('Using device: ',device)



# Creating Custom Dataclass
class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features= torch.tensor(features, dtype= torch.float32)   
        self.target= torch.tensor(target, dtype= torch.float32)
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
        

# Creating dataloaders
train_dataset= CustomDataset(X_train, y_train)
test_dataset= CustomDataset(X_test, y_test)

train_dataloader= DataLoader(train_dataset, shuffle= True, batch_size=6)
test_dataloader= DataLoader(test_dataset, shuffle= False, batch_size= 6)


# Training loop
model= Model(input_features= X_train.shape[1],
             hidden_neurons= 256,
             output_features= 1)

model.to(device= device)

loss_fn= nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(), lr= 3e-3)
writer= SummaryWriter()

epochs= 200
per_epoch_loss_list= []
for epoch in range(epochs):
    
    model.train()
    total_epoch_loss= 0
    for batch_features, batch_target in train_dataloader:
        
        batch_features, batch_target= batch_features.to(device), batch_target.to(device)
        y_pred= model(batch_features)
        loss= loss_fn(y_pred.view(-1), batch_target.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_epoch_loss += loss.item()
    avg_loss= total_epoch_loss/len(train_dataloader)
    per_epoch_loss_list.append(avg_loss)
    writer.add_scalar('Training Loss', avg_loss, epoch)
    
    print(f'Epoch: {epoch+1}, Avg. loss: {avg_loss}')
    
plt.plot(per_epoch_loss_list)
plt.title('Average training loss')
plt.ylabel('Avg. train loss')
plt.xlabel('Epochs')
plt.grid()
plt.savefig('Train loss curve.png')
print('plot saved !!')
plt.close()
        

# Evaluating the model
model.eval()
all_preds= []
all_targets= []

with torch.no_grad():
    for batch_features, batch_target in test_dataloader:
        batch_features, batch_target= batch_features.to(device), batch_target.to(device)
        
        preds= model(batch_features).view(-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_target.cpu().numpy())
    
all_preds= np.concatenate(all_preds)
all_targets= np.concatenate(all_targets)

mse= mean_squared_error(all_preds, all_targets)
mape= mean_absolute_percentage_error(all_preds, all_targets)

print('MSE: ', mse)
print('MAPE: ', mape)  