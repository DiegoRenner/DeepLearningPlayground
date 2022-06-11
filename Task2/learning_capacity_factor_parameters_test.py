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

# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

fname = os.path.join("TrainingData_1601.txt")
values_data = np.loadtxt(fname, delimiter=" ")
fname = os.path.join("samples_sobol.txt")
values_points = np.loadtxt(fname, delimiter=" ")

x = torch.from_numpy(values_points[0:values_data.shape[0],:]).float()
y = torch.from_numpy(values_data[:,8]).reshape((values_data.shape[0],1)).float()
n_samples = x.shape[0]


network_properties_final = {
    "hidden_layers": [2],
    "neurons": [5],
    "regularization_exp": [2],
    "regularization_param": [0],
    "batch_size": [n_samples],
    "epochs": [2500],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [134],
    "activation_function": ["Sigmoid"]
}

settings = list(itertools.product(*network_properties_final.values()))

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

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
sorted_indices = np.flip(np.argsort(val_err_conf))
print("Configurations from worst to best: ")
for i in sorted_indices:
    print("###################################", i, "###################################")
    print(settings[i])
    print(val_err_conf[i]**0.5*100)

fname = os.path.join("TestingData.txt")
output_dimension = 1
test_data = np.loadtxt(fname, delimiter=" ")
prediction = np.ndarray((test_data.shape[0], test_data.shape[1]+output_dimension))
input_dim = test_data.shape[1]
prediction[:,0:input_dim] = test_data
prediction = torch.from_numpy(prediction)
for i in np.arange(0,test_data.shape[0]):
    prediction[i,input_dim:input_dim+output_dimension] = model(prediction[i,0:input_dim].float())
prediction = prediction.detach().numpy()
fname = os.path.join("SubTask2-"+dt_string+".txt")
np.savetxt(fname, prediction[:,input_dim])
