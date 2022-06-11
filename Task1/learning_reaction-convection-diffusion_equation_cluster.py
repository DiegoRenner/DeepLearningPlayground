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
values = np.loadtxt(fname, skiprows=1, delimiter=",")
figure = plt.figure()
plt.plot(values[:,0],values[:,1])
plt.plot(values[:,0],values[:,2])
mean0 = np.mean(values[:,0])
mean1 = np.mean(values[:,1])
mean2 = np.mean(values[:,2])
max0 = np.max(values[:,0])
max1 = np.max(values[:,1])
max2 = np.max(values[:,2])
min0 = np.min(values[:,0])
min1 = np.min(values[:,1])
min2 = np.min(values[:,2])
values[:,0] = (values[:,0])/max0
values[:,1] = (values[:,1])/max1
values[:,2] = (values[:,2])/max2

n_samples = values.shape[0]
t = torch.from_numpy(values[:,0].transpose()).reshape((n_samples, 1)).float()
T0 = torch.from_numpy(values[:,1:3]).float()

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

    model, relative_error_train_, relative_error_val_= run_configuration(setup_properties, t, T0)
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
test_data = test_data.reshape(test_data.shape[0],1)/max0
input_dim = test_data.shape[1]
output_dimension = 2
prediction = np.ndarray((test_data.shape[0], input_dim+output_dimension))
prediction[:,0:input_dim] = test_data
prediction = torch.from_numpy(prediction)
for i in np.arange(0,test_data.shape[0]):
    prediction[i,input_dim:input_dim+output_dimension] = model(prediction[i,0:input_dim].float())
prediction = prediction.detach().numpy()

fname = os.path.join("SubTask1.txt")
print(prediction)
prediction[:,0] = prediction[:,0]*max0
prediction[:,1] = prediction[:,1]*max1
prediction[:,2] = prediction[:,2]*max2
figure = plt.figure()
plt.plot(prediction[:, 0], prediction[:, 1])
plt.plot(prediction[:, 0], prediction[:, 2])
plt.show()
np.savetxt(fname, prediction, header="t,tf0,ts0",delimiter=",")
