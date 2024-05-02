import torch
import torch.nn as nn
import snntorch as snn 
from snntorch import surrogate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch.jit
import coremltools

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)


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
    
    def linear_layer(input, weight, bias):
         return torch.matmul(input, weight) + bias
     
    def distances(x , y):
        #x = torch.from_numpy(x)
        discriminant = torch.sum((y - x)**2, dim = 1)
        distance = torch.sqrt(discriminant)
        return distance

    class RadialBasis(nn.Module):
        def __init__ (self, hidden):
              super().__init__()
              self.input = x_train.shape[1]
              self.hidden = hidden
              self.output = y_train.shape[1]
              self. device = device
              self.centre = nn.Parameter(torch.randn((1, self.input)))
              self.width = nn.Parameter(torch.randn((1,)))
              self.weight = nn.Parameter(torch.randn((self.hidden, self.output)))
              self.bias = nn.Parameter(torch.randn((1,)))
        
        def forward(self, x):
             distance = distances(x, self.centre)
             multiplier = torch.ones(1, self.hidden).float()

             def gaussian(distances, width, multiplier):
                phi = torch.exp(-(distances ** 2)/ ((2 * width) ** 2))
                phi = torch.reshape(phi,(x.shape[0]  , 1)).float()
                return torch.matmul(phi, multiplier) 
             
             activations = gaussian(distance, width = self.width, multiplier = multiplier)
             output = linear_layer(activations, weight = self.weight, bias = self.bias)
             return output

    model = RadialBasis(hidden = 10)

    iterations = 110
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    reg_strength = 0.001
    model.train()

    for epoch in range(iterations):
          for i, (x_train,y_train) in enumerate(traindataloader):
                         optimizer.zero_grad()
                         output = model(x_train)
                         cost = loss(y_train, output) 
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

model.eval()
example_input = torch.randn(1, 3, 224, 224)

# Convert the PyTorch model to TorchScript (jit) format
traced_model = torch.jit.trace(model, x_train)

# Convert the TorchScript (jit) model to CoreML format
coreml_model = coremltools.convert(traced_model, inputs=[coremltools.TensorType(name="input", shape=x_train.shape)])

# Save the converted CoreML model to disk
coreml_model.save("converted_model.mlmodel")


     

                         

    