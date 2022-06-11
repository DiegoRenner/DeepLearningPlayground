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
import sys
sys.path.insert(1, '..')
from Common_mod import NeuralNet, fit, run_configuration
from datetime import datetime

torch.set_num_threads(72)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")

fname = os.path.join("TrainingData.txt")
values = np.loadtxt(fname, skiprows=1, delimiter=",")

max1 = np.max(values[:,1])
max2 = np.max(values[:,2])
values[:,1] = (values[:,1])/max1
values[:,2] = (values[:,2])/max2
n_meas = values.shape[0]
sample_size = 50
n_samples = n_meas - sample_size
x = np.ndarray((n_samples,2*sample_size))
y = np.ndarray((n_samples,2))
for i in np.arange(0,n_samples):
    x[i,:] = values[i:i+sample_size,1:3].reshape(1,2*sample_size)
    y[i] = values[i+sample_size,1:3].reshape(1,2)

perm = np.random.permutation(n_samples)
x = torch.from_numpy(x[perm,:]).float()
y = torch.from_numpy(y[perm,:]).float()



# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

network_properties = {
    "hidden_layers": [8],
    "neurons": [10],
    "regularization_exp": [2],
    "regularization_param": [0],
    "batch_size": [n_samples],
    "epochs": [5000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [34],
    "activation_function": ["Tanh"]
}

settings = list(itertools.product(*network_properties.values()))

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

    model, relative_error_train_, relative_error_val_= run_configuration(setup_properties, x, y)
    if (len(val_err_conf)==0 or relative_error_val_ < min(val_err_conf)):
        print("Found best model so far! Saving it.")
        torch.save(model.state_dict(), "models/model_cluster-" + dt_string + ".pth")
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

fname = os.path.join("TestingData.txt")
test_data = np.loadtxt(fname, skiprows=1, delimiter=",")
test_data = test_data.reshape(test_data.shape[0], 1)
input_dim = test_data.shape[1]
output_dimension = 2
prediction = np.ndarray((test_data.shape[0], input_dim+output_dimension))
prediction[:,0:input_dim] = test_data
prediction = torch.from_numpy(prediction)
fname = os.path.join("TrainingData.txt")
test_data_base = np.loadtxt(fname, skiprows=1, delimiter=",")
test_data_base[:,1] /=max1
test_data_base[:,2] /=max2
base_len = 50
input = np.ndarray((base_len+test_data.shape[0], 2))
input[0:base_len,:] = test_data_base[-base_len:,1:3]
input = input[0:base_len,:].reshape(2*base_len,1).transpose()
input = torch.from_numpy(input)
for i in np.arange(0,test_data.shape[0]):
    prediction[i,input_dim:input_dim+output_dimension] = model(input.float())
    input = np.ndarray((base_len + test_data.shape[0], 2))
    input[0:base_len, :] = test_data_base[-base_len:, 1:3]
    input[base_len:base_len+i+1,:] = prediction[0:i+1,input_dim:input_dim+output_dimension].detach().numpy()
    if i == test_data.shape[0]-1:
        figure = plt.figure()
        plt.plot(input[:,0])
        plt.plot(input[:,1])
        plt.show()

    input = input[i+1:base_len+i+1, :].reshape(2 * base_len, 1).transpose()
    input = torch.from_numpy(input)
prediction = prediction.detach().numpy()
fname = os.path.join("SubTask3-"+dt_string+".txt")
prediction[:,1] = prediction[:,1]*max1
prediction[:,2] = prediction[:,2]*max2
np.savetxt(fname, prediction, header="t,tf0,ts0",delimiter=",")
