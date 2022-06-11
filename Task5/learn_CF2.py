import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
from Common import NeuralNet, fit

fname = os.path.join("TrainingData.txt")
data = np.loadtxt(fname, skiprows=0, delimiter=" ")
n_data = data.shape[0]

#n_mean_running_window = 10
#n_mean = n_measurements-n_mean_running_window
#n_std = n_mean
#mean = np.zeros((n_std, 2))
#std = np.zeros((n_std, 1))
#for i in range(0,n_std):
#    mean[i] = np.mean(measurements[i:i + n_mean_running_window, :], axis=0)
#    std[i] = np.sqrt(np.sum(np.power(mean[i,1]-measurements[i:i+n_mean_running_window,1],2)))
#
#print(mean)

#plt.plot(data[:, 0], data[:, 1], color="red")
#plt.plot(data[:, 0], std, color="red")
#plt.plot(measurements[:, 0], measurements[:, 2])
#np.random.shuffle(measurements)
#mean0 = np.mean(measurements[:, 0])
#mean1 = np.mean(measurements[:, 1])
#mean2 = np.mean(measurements[:, 2])
#max0 = np.max(data[:, 0])
#max1 = np.max(data[:, 1])
#max2 = np.max(data[:, 2])
#min0 = np.min(measurements[:, 0])
#min1 = np.min(measurements[:, 1])
#min2 = np.min(measurements[:, 2])
#data[:, 0] = (data[:, 0]) / max0
#data[:, 1] = (data[:, 1]) / max1
#data[:, 2] = (data[:, 2]) / max2

#t = torch.from_numpy(data[:, 0].transpose()).reshape((265, 1)).float()
input_data = torch.from_numpy(data[:, 0:2].transpose()).reshape((n_data,2)).float()
output_data = torch.from_numpy(data[:, 2].transpose()).reshape((n_data,1)).float()
#Tf0 = values[:,1]
#Ts0 = values[:,2]
#sigma = torch.from_numpy(data[:, 1].transpose()).reshape((n_data,1)).float()
batch_size = n_data
training_set_loader = DataLoader(torch.utils.data.TensorDataset(input_data, output_data), batch_size=batch_size, shuffle=True)

model = NeuralNet(input_dimension=input_data.shape[1],
                  output_dimension=output_data.shape[1],
                  n_hidden_layers=16,
                  neurons=400,
                  regularization_param=0.0,
                  regularization_exp=2,
                  retrain_seed=128)


optimizer_ = optim.LBFGS(model.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)

n_epochs = 2500
history = fit(model, training_set_loader, n_epochs, optimizer_, p=2,
        verbose=True)
print("Final Training loss: ", history[-1])

