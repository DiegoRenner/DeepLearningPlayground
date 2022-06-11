import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,
            neurons, regularization_param, regularization_exp,
            activation_function):
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


    def init_xavier(self, model, retrain_seed, activation_function):
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

    def regularization(self, model, p):
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
                #print("x:\n",x_train_,"\n u_train:\n",u_train_,"\n u_pred:\n",u_pred_)
                loss_u = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
                loss_reg = model.regularization(model, regularization_exp)
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


def run_configuration(conf_dict, x, y, validation_set=True):
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
    if not (validation_set):
        validation_size = 0
        training_size = x.shape[0]

    x_train = x[:training_size, :]
    y_train = y[:training_size, :]

    x_val = x[training_size:, :]
    y_val = y[training_size:, :]

    training_set = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    print(x.shape[1])
    print(y.shape[1])
    model = NeuralNet(input_dimension=x.shape[1],
                           output_dimension=y.shape[1],
                           n_hidden_layers=n_hidden_layers,
                           neurons=neurons,
                           regularization_param=regularization_param,
                           regularization_exp=regularization_exp,
                           activation_function=activation_function)

    # Xavier weight initialization
    model.init_xavier(model, retrain, activation_function)

    if opt_type == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=0.001)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(model.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
    else:
        raise ValueError("Optimizer not recognized")

    history = fit(model, training_set, x_val, y_val, n_epochs, optimizer_,
            p=2, verbose=False)

    y_val = y_val.reshape(-1, )
    y_train = y_train.reshape(-1, )

    y_val_pred = model(x_val).reshape(-1, )
    y_train_pred = model(x_train).reshape(-1, )

    # Compute the relative training error
    relative_error_train = torch.mean((y_train_pred - y_train) ** 2) / torch.mean(y_train ** 2)
    print("Relative Training Error: ", relative_error_train.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative validation error
    relative_error_val = torch.mean((y_val_pred - y_val) ** 2) / torch.mean(y_val ** 2)
    print("Relative Validation Error: ", relative_error_val.detach().numpy() ** 0.5 * 100, "%")

    prediction_t = x
    prediction_t_tensor = x#torch.from_numpy(prediction_t)

    return model, relative_error_train.item(), relative_error_val.item()

class Legendre(nn.Module):
    """ Univariate Legendre Polynomial """

    def __init__(self, PolyDegree):
        super(Legendre, self).__init__()
        self.degree = PolyDegree

    def legendre(self,x, degree):
        x = x.reshape(-1, 1)
        list_poly = lst()
        zeroth_pol = torch.ones(x.size(0),1)
        list_poly.append(zeroth_pol)
        # retvar[:, 0] = x * 0 + 1
        if degree > 0:
            first_pol = x
            list_poly.append(first_pol)
            ith_pol = torch.clone(first_pol)
            ith_m_pol = torch.clone(zeroth_pol)

            for ii in range(1, degree):
                ith_p_pol = ((2 * ii + 1) * x * ith_pol - ii * ith_m_pol) / (ii + 1)
                list_poly.append(ith_p_pol)
                ith_m_pol = torch.clone(ith_pol)
                ith_pol = torch.clone(ith_p_pol)
        list_poly = torch.cat(list_poly,1)
        return list_poly

    def forward(self, x):
        eval_poly = self.legendre(x, self.degree)
        return eval_poly

class MultiVariatePoly(nn.Module):

    def __init__(self, dim, order):
        super(MultiVariatePoly, self).__init__()
        self.order = order
        self.dim = dim
        self.polys = Legendre(order)
        self.num = (order + 1) ** dim
        self.linear = torch.nn.Linear(self.num, 1)

    def forward(self, x):
        poly_eval = list()
        leg_eval = torch.cat([self.polys(x[:, i]).reshape(1, x.shape[0], self.order + 1) for i in range(self.dim) ])
        for i in range(x.shape[0]):
            poly_eval.append(torch.torch.cartesian_prod(*leg_eval[:, i, :]).prod(dim=1).view(1, -1))
        poly_eval = torch.cat(poly_eval)
        return self.linear(poly_eval)
