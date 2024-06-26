import torch
import torch.nn as nn
import snntorch as snn 
from snntorch import surrogate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import sys
seed = 1
torch.manual_seed(seed)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class CustomData(Dataset):
     def __init__(self, features, labels, transform = None):
          self.features = features
          self.labels = labels
          self.transform = transform

     def __len__(self):
          return len(self.features)
     
     def __getitem__(self, i):
          samples = (self.features[i], self.labels[i])
          if self.transform:
               samples = self.transform(samples)
          return samples
               
class ToTensor():
     def __call__(self, samples):
          features, labels = samples
          return torch.from_numpy(features).float(), torch.from_numpy(labels).float()

if __name__ == "__main__":
    x_train_path = r"/Users/Jay/Desktop/dataset/x_train.xlsx"
    x_test_path = r"/Users/Jay/Desktop/dataset/x_test.xlsx"
    y_train_path = r"/Users/Jay/Desktop/dataset/y_train.xlsx"
    y_test_path =  r"/Users/Jay/Desktop/dataset/y_test.xlsx"

    x_train = pd.read_excel(x_train_path, header = None)
    x_test = pd.read_excel(x_test_path , header = None)
    y_train = pd.read_excel(y_train_path , header = None)
    y_test = pd.read_excel(y_test_path , header = None)

    x_train = torch.tensor(x_train.values).float()
    x_test = torch.tensor(x_test.values).float()
    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()
    
   
    
    traindataset = CustomData(x_train, y_train, transform = None)
    testdataset = CustomData(x_test, y_test, transform = None)

    traindataloader = DataLoader(dataset=traindataset, batch_size = 7, shuffle = False)
    testdataloader = DataLoader(dataset=testdataset, batch_size = 3, shuffle = False)

    class NeuralNet(nn.Module):
               def __init__(self, input = x_train.shape[1], output = y_train.shape[1]):
                    super().__init__()
                    self.fc = nn.Sequential(nn.Linear(input,5),
                                             nn.Sigmoid(),
                                             nn.Linear(5,output)
                                             )
                    
               def forward(self,x):
                    out = self.fc(x)
                    return out
                    
    model = NeuralNet()

    iterations = 691
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    reg_strength = 0.001
    model.train()
    
    preds = []

    for epoch in range(iterations):
          for i, (x_train,y_train) in enumerate(traindataloader):
                         abs_sum_weights = 0

                         for params in model.parameters():
                                abs = torch.abs(params)
                                abs_sum_weights += abs
                         lasso = reg_strength * torch.sum(abs_sum_weights)
                         lasso = lasso.clone()
                        
                         
                         optimizer.zero_grad()
                         output = model(x_train)
                         cost = loss(y_train, output) + lasso
                         cost.backward()
                         optimizer.step()

                         
          if epoch % 10 == 0:
                print(f"Epoch: {epoch + 1}, mse: {cost:.6f} ")
     
    model.eval()
    total_sample = 0
    total_loss = 0
    for i, (x_test,y_test) in enumerate(testdataloader):
                    test_prediction = model(x_test)
                    test_loss = loss(y_test, test_prediction)
                    total_loss += test_loss.item() * len(x_test)
                    total_sample += len(x_test)
                    average_loss = total_loss/total_sample
                    
                    

    print(f"Test mse: {average_loss:.6f}")
    
    average_loss = torch.tensor(average_loss)

    model.eval()
    with torch.no_grad():
               x_train_path = r"/Users/Jay/Desktop/dataset/x_train.xlsx"
               x_test_path = r"/Users/Jay/Desktop/dataset/x_test.xlsx"
               

               x_train = pd.read_excel(x_train_path, header = None)
               x_test = pd.read_excel(x_test_path , header = None)
              
               x_train = torch.tensor(x_train.values).float()
               x_test = torch.tensor(x_test.values).float()
               
               train_pred = model(x_train)
               test_pred = model(x_test)




sys.exit()
path1 = "/Users/Jay/Desktop/dataset/bpnn_train.xlsx"
path2 = "/Users/Jay/Desktop/dataset/bpnn_test.xlsx"

pd1 = pd.DataFrame(train_pred)
pd2 = pd.DataFrame(test_pred)

pd1.to_excel(path1, index=None, header=None)
pd2.to_excel(path2, index=None, header=None)
