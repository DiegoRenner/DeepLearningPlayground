import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import itertools


class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, activation_function):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        if activation_function=="Tanh":
            self.activation = nn.Tanh()
        elif activation_function=="ReLU":
            self.activation = nn.ReLU()
        elif activation_function=="Sigmoid":
            self.activation = nn.Sigmoid()

        #
        self.regularization_param = regularization_param
        #
        self.regularization_exp = regularization_exp

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)


def init_xavier(model, retrain_seed,activation_function):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if activation_function == "Tanh":
                g = nn.init.calculate_gain('tanh')
            elif activation_function == "ReLU":
                g = nn.init.calculate_gain('relu')
            elif activation_function == "Sigmoid":
                g = nn.init.calculate_gain('sigmoid')

            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def fit(model, training_set, x_validation_, y_validation_, num_epochs, optimizer, p, verbose=True):
    history = [[], []]
    regularization_param = model.regularization_param
    regularization_exp = model.regularization_exp

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                loss_u = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item() / len(training_set)
                return loss

            optimizer.step(closure=closure)

        y_validation_pred_ = model(x_validation_)
        validation_loss = torch.mean((y_validation_pred_.reshape(-1, ) - y_validation_.reshape(-1, )) ** p).item()
        history[0].append(running_loss[0])
        history[1].append(validation_loss)

        if verbose:
            print('Training Loss: ', np.round(running_loss[0], 8))
            print('Validation Loss: ', np.round(validation_loss, 8))

    print('Final Training Loss: ', np.round(history[0][-1], 8))
    print('Final Validation Loss: ', np.round(history[1][-1], 8))
    return history


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
    y_train = y[:training_size]

    x_val = x[training_size:, :]
    y_val = y[training_size:]

    training_set = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    my_network = NeuralNet(input_dimension=x.shape[1],
                           output_dimension=y.shape[1],
                           n_hidden_layers=n_hidden_layers,
                           neurons=neurons,
                           regularization_param=regularization_param,
                           regularization_exp=regularization_exp,
                           activation_function=activation_function)

    # Xavier weight initialization
    init_xavier(my_network, retrain, activation_function)

    if opt_type == "ADAM":
        optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
    else:
        raise ValueError("Optimizer not recognized")

    history = fit(my_network, training_set, x_val, y_val, n_epochs, optimizer_, p=2, verbose=False)

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

    fname = os.path.join("Task2/TestingData.txt")
    test_data = np.loadtxt(fname, delimiter=" ")
    prediction = np.ndarray((test_data.shape[0], 9))
    input_dim = test_data.shape[1]
    prediction[:,0:input_dim] = test_data
    prediction = torch.from_numpy(prediction)
    for i in np.arange(0,test_data.shape[0]):
        prediction[i,input_dim] = my_network(prediction[i,0:input_dim].float())
    prediction = prediction.detach().numpy()
    fname = os.path.join("Task2/SubTask2.txt")
    np.savetxt(fname, prediction[:,input_dim])



    return relative_error_train.item(), relative_error_val.item()


# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

fname = os.path.join("Task2/TrainingData_1601.txt")
values_data = np.loadtxt(fname, delimiter=" ")
fname = os.path.join("Task2/samples_sobol.txt")
values_points = np.loadtxt(fname, delimiter=" ")

#transformations_data = list()
#for i in np.arange(0,9):
#    values_data[i,:] = (values_data[i,:])/np.max(values_data[i,:])
#    transformations_data = (np.mean(values_data[i,:]), np.max(values_data[i,:]))
#
#transformations_points = list()
#for i in np.arange(0,8):
#    values_points[i,:] = (values_points[i,:])/np.max(values_points[i,:])
#    transformations_points = (np.mean(values_points[i,:]), np.max(values_points[i,:]))
y = torch.from_numpy(values_data[:,8]).reshape((values_data.shape[0],1)).float()
#Tf0 = values[:,1]
#Ts0 = values[:,2]
x = torch.from_numpy(values_points[0:values_data.shape[0],:]).float()
n_samples = x.shape[0]
batch_size = n_samples
training_set = DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True)


network_properties_final = {
    "hidden_layers": [1],
    "neurons": [1],
    "regularization_exp": [2],
    "regularization_param": [0],
    "batch_size": [n_samples],
    "epochs": [10],
    "optimizer": ["ADAM"],
    "init_weight_seed": [567],
    "activation_function": ["Tanh"]
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

    relative_error_train_, relative_error_val_= run_configuration(setup_properties, x, y)
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
