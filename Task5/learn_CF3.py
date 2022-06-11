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
max0 = np.max(data[:, 0])
max1 = np.max(data[:, 1])
#max2 = np.max(data[:, 2])
#min0 = np.min(measurements[:, 0])
#min1 = np.min(measurements[:, 1])
#min2 = np.min(measurements[:, 2])
#data[:, 0] = (data[:, 0]-2) / 18
#data[:, 1] = (data[:, 1]-50) / 350
#data[:, 2] = (data[:, 2]) / max2

#t = torch.from_numpy(data[:, 0].transpose()).reshape((265, 1)).float()
input_data = torch.from_numpy(data[:, 0:2]).reshape((n_data,2)).float()
output_data = torch.from_numpy(data[:, 2].transpose()).reshape((n_data,1)).float()
plt.scatter(input_data[:,0], input_data[:,1], c=output_data)
plt.show()
#Tf0 = values[:,1]
#Ts0 = values[:,2]
#sigma = torch.from_numpy(data[:, 1].transpose()).reshape((n_data,1)).float()
batch_size = n_data




def run_configuration(conf_dict, x, y):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print(conf_dict)

    # Get the confgiuration to test
    opt_type = conf_dict["optimizer"]
    n_epochs = conf_dict["epochs"]
    n_hidden_layers = conf_dict["hidden_layers"]
    neurons = conf_dict["neurons"]
    regularization_param = conf_dict["regularization_param"]
    regularization_exp = conf_dict["regularization_exp"]
    retrain = conf_dict["init_weight_seed"]
    batch_size = conf_dict["batch_size"]
    activation_function = conf_dict["activation_function"]

    validation_size = int(20 * x.shape[0] / 100)
    training_size = x.shape[0] - validation_size
    x_train = x[:training_size, :]
    y_train = y[:training_size, :]

    x_val = x[training_size:, :]
    y_val = y[training_size:, :]

    training_set = DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size, shuffle=False)


    my_network = NeuralNet(input_dimension=x.shape[1],
                           output_dimension=y.shape[1],
                           n_hidden_layers=n_hidden_layers,
                           neurons=neurons,
                           regularization_param=regularization_param,
                           regularization_exp=regularization_exp,
                            retrain_seed=128)
    print(my_network)

    # Xavier weight initialization
    #init_xavier(my_network, retrain, activation_function)

    if opt_type == "ADAM":
        optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
    else:
        raise ValueError("Optimizer not recognized")

    history = fit(my_network, training_set,  n_epochs, optimizer_, p=2,
            verbose=True)

    y_val = y_val.reshape(-1, )
    y_train = y_train.reshape(-1, )

    y_val_pred = my_network(x_val).reshape(-1, )
    y_train_pred = my_network(x_train).reshape(-1, )

    # Compute the relative training error
    relative_error_train = torch.mean((y_train_pred - y_train) ** 2) / torch.mean(y_train ** 2)
    print("Relative Training Error: ", relative_error_train.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative validation error
    relative_error_val = torch.mean((y_val_pred - y_val) ** 2) / torch.mean(y_val ** 2)
    print("Relative Validation Error: ", relative_error_val.detach().numpy() ** 0.5 * 100, "%")

    #fname = os.path.join("Task1/TestingData.txt")
    #test_data = np.loadtxt(fname, skiprows=1, delimiter=",")
    #test_data = test_data.reshape(test_data.shape[0],1)#/max0
    #input_dim = test_data.shape[1]
    #output_dimension = 2
    #prediction = np.ndarray((test_data.shape[0], input_dim+output_dimension))
    #prediction[:,0:input_dim] = test_data
    #prediction = torch.from_numpy(prediction)
    #for i in np.arange(0,test_data.shape[0]):
    #    prediction[i,input_dim:input_dim+output_dimension] = my_network(prediction[i,0:input_dim].float())
    #prediction = prediction.detach().numpy()

    #fname = os.path.join("Task1/SubTask1.txt")
    #print(prediction)
    #prediction[:,0] = prediction[:,0]#*max0
    #prediction[:,1] = prediction[:,1]#*max1
    #prediction[:,2] = prediction[:,2]#*max2
    #figure = plt.figure()
    #plt.plot(prediction[:, 0], prediction[:, 1])
    #plt.plot(prediction[:, 0], prediction[:, 2])
    #plt.show()
    #np.savetxt(fname, prediction, header="t,tf0,ts0",delimiter=",")
    prediction_t = input_data
    prediction_t_tensor = input_data
    #torch.from_numpy(prediction_t)
    #prediction_sigma = np.arange(0,0.25,0.001)

    for i in np.arange(0,prediction_t.shape[0],1):
        print(prediction_t_tensor[i].reshape(1,2).float())
        print(my_network(prediction_t_tensor[i].reshape(1,2).float()))
    ##    prediction_sigma[i] = my_network(prediction_t_tensor[i].reshape(1,1).float()).detach().numpy()
    ##figure = plt.figure()
    ##plt.scatter(measurements[:, 0], measurements[:, 1], marker="o")
    ##plt.plot(prediction_t,prediction_sigma, color="red")
    ##plt.show()
    #print(my_network(my_network(prediction.float())))

    return relative_error_train.item(), relative_error_val.item()


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
    "hidden_layers": [20],
    "neurons": [100],
    "regularization_exp": [2],
    "regularization_param": [0.0],
    "batch_size": [n_data], #batch_size=1 to expensive
    "epochs": [1000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567],
    "activation_function": ["ReLU"]
}
network_properties_debug = {
    "hidden_layers": [8],
    "neurons": [20],
    "regularization_exp": [2],
    "regularization_param": [0.0001],
    "batch_size": [n_data],
    "epochs": [100],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567,122,231],
    "activation_function": ["ReLU"]
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
    #print("input_data:\n", input_data)
    relative_error_train_, relative_error_val_= run_configuration(setup_properties, input_data, output_data)
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


#plt.figure(figsize=(16, 8))
#plt.grid(True, which="both", ls=":")
#plt.scatter(np.log10(train_err_conf), np.log10(test_err_conf), marker="*", label="Training Error")
#plt.scatter(np.log10(val_err_conf), np.log10(test_err_conf), label="Validation Error")
#plt.xlabel("Selection Criterion")
#plt.ylabel("Generalization Error")
#plt.title(r'Validation - Training Error VS Generalization error ($\sigma=0.0$)')
#plt.legend()
#plt.savefig("sigma.png", dpi=400)
#plt.show()
