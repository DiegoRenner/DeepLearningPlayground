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
data = np.loadtxt(fname, skiprows=0, delimiter=" ")
n_measurements = data.shape[0]

n_samples = data.shape[0]
t = torch.from_numpy(data[:, 0:2]).reshape((n_samples,2)).float()
measurement_at_t = torch.from_numpy(data[:, 2].transpose()).reshape((n_samples,1)).float()


# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

network_properties = {
    "hidden_layers": [2, 4, 8],
    "neurons": [5, 10, 20],
    "regularization_exp": [2],
    "regularization_param": [0, 1e-4],
    "batch_size": [int(np.floor(n_samples/10)), int(np.floor(n_samples/2)), n_samples], #batch_size=1 to expensive
    "epochs": [1000, 2500, 5000],
    "optimizer": ["ADAM", "LBFGS"],
    "init_weight_seed": [567, 34, 134],
    "activation_function": ["Tanh", "ReLU", "Sigmoid"]
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
    model, relative_error_train_, relative_error_val_= run_configuration(setup_properties, t, measurement_at_t, validation_set=False)
    if (len(val_err_conf)==0 or relative_error_val_ < min(val_err_conf)):
        print("Found best model so far! Saving it.")
        torch.save(model.state_dict(), "models/b_model1-" + dt_string + ".pth")
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
sorted_indices = np.flip(np.argsort(val_err_conf))
print("Configurations from worst to best: ")
for i in sorted_indices:
    print("###################################", i, "###################################")
    print(settings[i])


