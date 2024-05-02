import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import sys

seed = 3
torch.manual_seed(seed)

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

    x_train = torch.tensor(x_train.values).float()
    x_test = torch.tensor(x_test.values).float()
    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()
    
    def sigmoid(x):
         activation = 1/(1 + torch.exp(-x))
         return activation
    
    def hidden_comp(x, y):
        return torch.matmul(x,y)
    
    def random(y, hidden):
        return torch.randn((y, hidden)) * math.sqrt(2.0 / (y + hidden))

    hidden = 2

    random_weights = random(x_train.shape[1], hidden) # Generates random weight matrix using the random function
    hidden_matrix = hidden_comp(x_train, random_weights) # Undertakes the input-hidden layer matrix computation using the hidden comp function
    hidden_output = sigmoid(hidden_matrix)
    pseudo_inverse = torch.linalg.pinv(hidden_output)
    optimal_weights = torch.matmul(pseudo_inverse, y_train)
    
    class ELM(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden

        def forward(self, x):
            rand = random_weights
            hidden_mat = hidden_comp(x, rand)
            hidden_out = sigmoid(hidden_mat)
            predict = torch.matmul(hidden_out,optimal_weights)
            return predict
        
    model = ELM(hidden = hidden)
    train_pred = model(x_train)
    test_pred = model(x_test)

    loss = nn.MSELoss()
    train_cost = loss(y_train, train_pred)
    test_cost = loss(y_test, test_pred)

    train_rmse = torch.sqrt(train_cost)
    test_rmse = torch.sqrt(test_cost)

    print(f"Train Cost: {train_cost:.6f}, rmse: {train_rmse:.6f}")
    print(f"Test cost: {test_cost:.6f}, rmse: {test_rmse:.6f}")

    print(train_pred)
    print(test_pred)






    

    