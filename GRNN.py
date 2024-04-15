import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys

seed = 0
torch.manual_seed(seed)

class CustomDataset (Dataset):
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
    path = r"/Users/Jay/Desktop/dataset/h2_data.csv"
    data = pd.read_csv(path)
    data = data.values
    x_data = data[ : , : -1]
    y_data = data[ : , [-1]]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, shuffle = True, random_state= seed )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_dataset = CustomDataset(x_train, y_train, transform= ToTensor())
    test_dataset = CustomDataset(x_test, y_test, transform= ToTensor())

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle= True)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size = 64)

    i = 0
    j = 0

    def euclidean_distance(x):
        return torch.sqrt(torch.sum(x))

    def point_distances(x,xk):
        distance_list = []
        for i in range(len(x)):
            for j in range(len(xk)):
                discriminant = (x[i] - xk[j]) ** 2
                distances = euclidean_distance(discriminant)
                distance_list.append(distances)
                j += 1
            i += 1
        centers = torch.tensor(distance_list)
        centers = centers.reshape(-1, 1)
        return centers
    
    def gaussian_kernel(x, width):
        gaussian = torch.exp(-x/(2 * (width**2)))
        return gaussian
    
    def sum(x, y):
      iterations = int(len(x))
      iterations_list = []
      counter = 0
      for i in range(iterations):
            counter_1 = counter + len(y)
            summation = torch.sum(x[counter : counter_1, : ])
            iterations_list.append(summation)
            counter += len(y)
            i += int(len(x)/len(y))
      iterations_list = torch.tensor(iterations_list, requires_grad=True)
      iterations_list = iterations_list[iterations_list != 0]
      iterations_list = iterations_list.reshape(-1,1)
      return iterations_list

    def output(x,y):
        return x/y     
    
    class GRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = x_train.shape[1]
            self.output = y_train.shape[1]
            self.width = nn.Parameter(torch.randn(1,), requires_grad = True)

        def forward(self, x):
            neurons = torch.ones_like(x)
            pattern_layer = x * neurons
            pseudo_centers = point_distances(x, pattern_layer)
            activations = gaussian_kernel(pseudo_centers, width = self.width)
            activations = activations.reshape(-1,1)
            activation_shape = (activations.shape)
            self.weight = nn.Parameter(torch.randn((activation_shape)), requires_grad = True)
            weighted_activations = activations * self.weight
            summation_weighted_activations = sum(weighted_activations, x)
            summation_activations = sum(activations, x)
            prediction = output(summation_weighted_activations , summation_activations)
            
            prediction = prediction.reshape(-1, 1)

            return prediction
            
    model = GRNN()
    x_train = torch.tensor(x_train)
    
    iterations = 941
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)
    loss = nn.MSELoss()

    for epoch in range(iterations):
        for i, (x_train, y_train) in enumerate(train_dataloader):
            prediction = model(x_train)
            cost = loss(y_train, prediction)
            rmse = torch.sqrt(cost)
            cost.backward()
            optimiser.step()
            optimiser.zero_grad()
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1}, loss: {cost:.4f}, rmse: {rmse:.4f}")

            
    total_cost = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_dataloader):
              test_prediction = model(x_test)
              total_cost += loss(y_test, test_prediction)
              total_samples += len(x_test)
              average_mse = total_cost/total_samples
              test_rmse = torch.sqrt(average_mse)
        print(f"The mse for the test set is: {average_mse:.4f}, and rmse is {test_rmse:.4f}")

        train_prediction = model(x_train)

        def pearson(x,y):
              mean_true = torch.mean(y)
              mean_pred = torch.mean(x)
              pred_subtract_mean = x - mean_pred
              true_subtract_mean = y - mean_true
              square_pred = pred_subtract_mean ** 2
              square_true = true_subtract_mean ** 2
              sum = torch.sum((pred_subtract_mean * true_subtract_mean))
              sum_square_pred = torch.sum(square_pred)
              sum_square_true = torch.sum(square_true)
              product_sum_squares = sum_square_pred * sum_square_true
              return sum/torch.sqrt(product_sum_squares)
            

        def mape(y_pred, y_true):
            return torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100

        def mae(y_pred, y_true):
            return torch.mean(torch.abs(y_true - y_pred)).item()


        def mae(y_pred, y_true):
            return torch.mean(torch.abs(y_true - y_pred)).item()

    print()
    train_r = pearson(train_prediction, y_train)
    print(f"The pearson correlation (r) for train is {train_r:.4f}")
    print()
    test_r = pearson(test_prediction, y_test)
    print(f"The pearson correlation (r) for test is {test_r:.4f}")
    print()
   
    print()
    train_mape = mape(train_prediction, y_train)
    print(f"The train mape is {train_mape:.4f}")
    print()
    test_mape = mape(test_prediction, y_test)
    print(f"The test mape is {test_mape:.4f}")
    print()
    train_mae = mae(train_prediction, y_train)
    print(f"The train mae is {train_mae:.4f}")
    print()
    test_mae = mae(test_prediction, y_test)
    print(f"The test mae is {test_mae:.4f}")
    print()
    
   
         
