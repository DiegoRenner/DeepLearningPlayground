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
from Common_mod import NeuralNet, fit, run_configuration
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")

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
max0 = np.max(data[:, 0])
max1 = np.max(data[:, 1])
#max2 = np.max(data[:, 2])
#min0 = np.min(measurements[:, 0])
#min1 = np.min(measurements[:, 1])
#min2 = np.min(measurements[:, 2])
data[:, 0] = (data[:, 0]) / max0
data[:, 1] = (data[:, 1]) / max1
#data[:, 2] = (data[:, 2]) / max2

#t = torch.from_numpy(data[:, 0].transpose()).reshape((265, 1)).float()
input_data = torch.from_numpy(data[:, 0:2]).reshape((n_data,2)).float()
output_data = torch.from_numpy(data[:, 2].transpose()).reshape((n_data,1)).float()
#Tf0 = values[:,1]
#Ts0 = values[:,2]
#sigma = torch.from_numpy(data[:, 1].transpose()).reshape((n_data,1)).float()
batch_size = n_data


# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)


#network_properties_final = {
#    "hidden_layers": [4],
#    "neurons": [20],
#    "regularization_exp": [2],
#    "regularization_param": [0.0004],
#    "batch_size": [n_data],
#    "epochs": [10000],
#    "optimizer": ["LBFGS"],
#    "init_weight_seed": [567],
#    "activation_function": ["Tanh"]
#}
network_properties = {
    "hidden_layers": [2, 4, 8],
    "neurons": [5, 10, 20],
    "regularization_exp": [2],
    "regularization_param": [0,1e-4],
    "batch_size": [int(np.floor(n_data/10)), int(np.floor(n_data/2)), n_data], #batch_size=1 to expensive
    "epochs": [1000, 2500, 5000],
    "optimizer": ["ADAM", "LBFGS"],
    "init_weight_seed": [567, 34, 134],
    "activation_function": ["Tanh", "ReLU", "Sigmoid"]
}
network_properties_debug = {
    "hidden_layers": [8],
    "neurons": [20],
    "regularization_exp": [2],
    "regularization_param": [0.0],
    "batch_size": [n_data],
    "epochs": [1000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567],
    "activation_function": ["Sigmoid"]
}

settings = list(itertools.product(*network_properties_debug.values()))

i = 0

train_err_conf = list()
val_err_conf = list()
print("Configurations to be run: ", len(settings))
for set_num, setup in enumerate(settings):
    print("###################################", set_num, "###################################")
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "regularization_exp": setup[2],
        "regularization_param": setup[3],
        "batch_size": setup[4],
        "epochs": setup[5],
        "optimizer": setup[6],
        "init_weight_seed": setup[7],
        "activation_function": setup[8]
    }
    #print("input_data:\n", input_data)
    model, relative_error_train_, relative_error_val_= run_configuration(setup_properties, input_data, output_data)
    if (len(val_err_conf)==0 or relative_error_val_ < min(val_err_conf)):
        print("Found best model so far! Saving it.")
        torch.save(model.state_dict(), "models/model1-" + dt_string + ".pth")
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)

print(train_err_conf, val_err_conf)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
sorted_indices = np.flip(np.argsort(val_err_conf))
print("Configurations from worst to best: ")
for i in sorted_indices:
    print("###################################", i, "###################################")
    print(settings[i])
    print(val_err_conf[i]**0.5*100)


