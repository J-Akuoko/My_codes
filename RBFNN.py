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

seed = 40
torch.manual_seed(seed)
np.random.seed(seed)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == "__main__":
    x_train_path = r"/Users/Jay/Desktop/dataset/x_train.xlsx"
    x_test_path = r"/Users/Jay/Desktop/dataset/x_test.xlsx"
    y_train_path = r"/Users/Jay/Desktop/dataset/y_train.xlsx"
    y_test_path =  r"/Users/Jay/Desktop/dataset/y_test.xlsx"

    x_train = pd.read_excel(x_train_path, header = None)
    x_test = pd.read_excel(x_test_path , header = None)
    y_train = pd.read_excel(y_train_path , header = None)
    y_test = pd.read_excel(y_test_path , header = None)

    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values

    data = [x_train, x_test, y_train, y_test]
    tensor_data = []

    for data in data:
          data = torch.from_numpy(data).float()
          tensor_data.append(data)

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
              self.input = tensor_data[0].shape[1]
              self.hidden = hidden
              self.output = tensor_data[2].shape[1]
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

    loss = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.01)
    iterations = 4000

    losses = []
    iteration = []

    for epoch in range(iterations):
             prediction = model(tensor_data[0])
             cost = loss(tensor_data[2], prediction)
             rmse = torch.sqrt(cost)
             cost.backward()
             opt.step()
             opt.zero_grad()

             if epoch % 10 == 0:
                print(f"Epoch: {epoch + 1}, mse:{cost:.6f}, rmse: {rmse:.6f}")

    losses.append(cost.item())
    iteration.append(epoch)

    total_cost = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
              test_prediction = model(tensor_data[1])
              total_cost += loss(tensor_data[3], test_prediction)
              total_samples += len(x_test)
              average_mse = total_cost/total_samples
              test_rmse = torch.sqrt(average_mse)
    print()
    print(f"The mse for the test set is: {average_mse:.6f}, and rmse is {test_rmse:.6f}")

    train_prediction = model(tensor_data[0])

    def mae(y_pred, y_true):
            return torch.mean(torch.abs(y_true - y_pred)).item()

    def nrmse(predictions, targets):
        
            # Calculate RMSE
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))

            # Calculate range of target variable
            target_range = torch.max(targets) - torch.min(targets)

            # Calculate NRMSE
            nrmse = rmse / target_range

            return nrmse.item()  # Convert to Python float

    train_mae = mae(train_prediction, tensor_data[2])
    print(f"The train mae is {train_mae:.4f}")
    print()
    test_mae = mae(test_prediction, tensor_data[3])
    print(f"The test mae is {test_mae:.4f}")
    print()
    train_nrmse = nrmse(train_prediction, tensor_data[2])
    print(f"The train nrmse is {train_nrmse:.4f}")
    print()
    test_nrmse = nrmse(test_prediction, tensor_data[3])
    print(f"The test nrmse is {test_nrmse:.4f}")
        
    tensor_data[2] = tensor_data[2].detach()
    tensor_data[3] = tensor_data[3].detach()
    test_prediction = test_prediction.detach()
    train_prediction = train_prediction.detach()

    def pearson_correlation(x, y):
    # Convert input lists to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Compute mean of x and y
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Compute covariance and variance
        cov_xy = np.sum((x - mean_x) * (y - mean_y))
        var_x = np.sum((x - mean_x) ** 2)
        var_y = np.sum((y - mean_y) ** 2)
        
        # Compute Pearson correlation coefficient
        corr_coef = cov_xy / np.sqrt(var_x * var_y)
        
        return corr_coef

    print("Train Pearson correlation coefficient:", pearson_correlation(train_prediction, tensor_data[2]).item())
    print("Test Pearson correlation coefficient:", pearson_correlation(test_prediction, tensor_data[3]).item())
    
    def r_squared(y_true, y_pred):
        # Mean of the true values
        mean_y_true = sum(y_true) / len(y_true)
        
        # Total sum of squares
        total_ss = sum((y_true - mean_y_true) ** 2)
        
        # Residual sum of squares
        residual_ss = sum((y_true - y_pred) ** 2)
        
        # Coefficient of determination (R-squared)
        r_squared = 1 - (residual_ss / total_ss)
        
        return r_squared

    print("Train r_squared is:", r_squared(tensor_data[2], train_prediction).item())
    print("Test r_squared is:", r_squared(tensor_data[3], test_prediction).item())
   

    sys.exit()
    path_1 = "/Users/Jay/Desktop/dataset/BPNN_train.xlsx"
    path_2 = "/Users/Jay/Desktop/dataset/BPNN_test.xlsx"

    df1 = pd.DataFrame(train_prediction.detach())
    df1.to_excel(path_1)

    df2 = pd.DataFrame(test_prediction.detach())
    df2.to_excel(path_2)
    


         