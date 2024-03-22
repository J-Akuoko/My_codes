"""
MIT License

Copyright (c) 2024 [Jeffrey Akuoko]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software...
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


seed = 50 # Setting seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available(): # GPU access 
    device = "cuda"
else:
    device = "cpu"

print("**** Crop yield prediciton using Spiking Neural Networks ****")


if __name__ == "__main__":
    path = "/Users/Jay/Desktop/crop.xlsx" # Data loading step
    data = pd.read_excel(path)
    data = data.values
    x_data = data[:,:-1]
    y_data = data[:,[-1]]

    x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size= 0.2, shuffle= True, random_state= seed) # Data splitting step
    scaler = MinMaxScaler() # Data processing step (Input features scaled between 0 and 1)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    class Custom(Dataset): # Creating a custom dataset
        def __init__(self, features, labels, transform = None):
            self.features = features
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.features)
        
        def __getitem__(self,i):
            sample = self.features[i], self.labels[i]
            if self.transform:
                sample = self.transform(sample)
            return sample
        
    class ToTensor(): # Creating a class for numpy conversion
        def __call__(self,sample):
            features, labels = sample
            return torch.from_numpy(features).float().to(device), torch.from_numpy(labels).float().to(device)

    train_dataset = Custom(x_train,y_train, transform=ToTensor()) # Creating an instance of the dataset class for the train data
    test_dataset = Custom(x_test,y_test, transform=ToTensor()) # Creating an instance of the dataset class for the test data

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0) # Data loading
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)

    class NeuralNet(nn.Module): # Creating the neural network architecture
        def __init__(self, input = x_train.shape[1], output = y_train.shape[1]):
            super().__init__()
            spk_grad = surrogate.fast_sigmoid()
            self.fc1 = nn.Linear(input,15)
            self.fc2 = nn.Linear(15,20)
            self.fc3 = nn.Linear(20,output)
            self.lif1 = snn.Leaky(beta = 0.25, learn_beta=True, learn_threshold=True, spike_grad=spk_grad)
            self.lif2 = snn.Leaky(beta = 0.25, learn_beta=True, learn_threshold=True, spike_grad=spk_grad)
            self.lif3 = snn.Leaky(beta = 0.25, learn_beta=True, learn_threshold=True,reset_mechanism="none")

        def forward(self,x): # Forward pass through the network
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            rec_mem3 = []
            simulation_steps = 10

            for step in range(simulation_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2,mem2)
                cur3 = self.fc3(spk2)
                _, mem3 = self.lif3(cur3,mem3)
                rec_mem3.append(mem3)
            stacked_mems = torch.stack(rec_mem3)
            stacked_mems = torch.sum(stacked_mems, dim = 0)
            return stacked_mems
    
    model = NeuralNet().to(device) 

    epochs = 271
    loss = nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    losses = []
    iterations = []

    # Training step
    model.train()
    for epoch in range(epochs):
        for i, (x_train,y_train) in enumerate(train_dataloader):
            train_predictions = model(x_train)
            cost = loss(y_train,train_predictions)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(cost.item())
        iterations.append(epoch)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1}, loss: {cost:.3f}")
    
     # Evaluation step
    model.eval()
    total_samples = 0
    test_losses = 0

    with torch.no_grad():
        for i, (x_test,y_test) in enumerate(test_dataloader):
            test_predictions = model(x_test)
            test_loss = loss(y_test,test_predictions)
            test_losses += test_loss
            total_samples += x_test.shape[0]
            mse_average = test_losses/total_samples
        print(f"The MSE for the test set is: {mse_average}")

    # Plotting the loss dynamics on the training set
    plt.plot(iterations,losses, label = "Train loss dynamics")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.title("Graph of loss dynamics on train set")
    plt.show()

#torch.save(model.state_dict(),"/Users/Jay/Desktop/SNN.pth")