import seaborn as sns
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC
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

torch.set_num_threads(16)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
dirstr = "run_b_" + dt_string
os.mkdir(dirstr)

fname = os.path.join("TrainingData.txt")
data = np.loadtxt(fname, skiprows=0, delimiter=" ")
n_measurements = data.shape[0]

n_samples = data.shape[0]
t = torch.from_numpy(data[:, 0:2]).reshape((n_samples,2)).float()
measurement_at_t = torch.from_numpy(data[:, 2].transpose()).reshape((n_samples,1)).float()


# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

network_properties1 = {
    "hidden_layers": [2, 4, 8],
    "neurons": [5, 10, 20],
    "regularization_exp": [2],
    "regularization_param": [0],
    "batch_size": [n_samples], #batch_size=1 to expensive
    "epochs": [2500, 5000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567],
    "activation_function": ["Tanh"]
}
network_properties = {
    "hidden_layers": [4],
    "neurons": [40],
    "regularization_exp": [2],
    "regularization_param": [0],
    "batch_size": [n_samples], #batch_size=1 to expensive
    "epochs": [5000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567],
    "activation_function": ["Sigmoid"]
}
network_properties_debug = {
    "hidden_layers": [2],
    "neurons": [5],
    "regularization_exp": [2],
    "regularization_param": [0],
    "batch_size": [n_samples], #batch_size=1 to expensive
    "epochs": [1000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [134],
    "activation_function": ["Tanh"]
}

settings = list(itertools.product(*network_properties.values()))

i = 0

train_err_conf = list()
val_err_conf = list()
best_model = NeuralNet(1,1,1,1,1,1,1)
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
    model, relative_error_train_, relative_error_val_= run_configuration(setup_properties, t, measurement_at_t)
    if (len(val_err_conf)==0 or relative_error_val_ < min(val_err_conf)):
        print("Found best model so far! Saving it.")
        best_model = model
        torch.save(model.state_dict(), dirstr+"/model.pth")
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
sorted_indices = np.flip(np.argsort(val_err_conf))
fname = os.path.join(dirstr+"/config.txt")
with open(fname, "a") as config:
    config.write("Configurations from worst to best: ")
    for i in sorted_indices:
        config.write("###################################"+  str(i) + "###################################")
        config.write(str(settings[i])+"\n")
        config.write(str(train_err_conf[i]) +", "+ str(val_err_conf[i])+"\n")


fname = os.path.join("MeasuredData.txt")
measurements = np.loadtxt(fname, skiprows=0, delimiter=" ")
n_measurements = measurements.shape[0]
data = measurements
n_samples = data.shape[0]
t = torch.from_numpy(data[:, 0].transpose()).reshape((n_samples,1)).float()
measurement_at_t = torch.from_numpy(data[:, 1].transpose()).reshape((n_samples,1)).float()
#print(t)
#print(measurement_at_t)

def G(inp):
    # Corresponds to G(x, w) from the notes, exact underlying model
    # this might come from the solution of PDE, for example
    return best_model(inp)
def G_sigma(inp):
    # Corresponds to G(x, w) from the notes, exact underlying model
    # this might come from the solution of PDE, for example
    x = inp[:,0]
    w = inp[:,1]
    #print(x,w,w*x)
    return (w*x).reshape((inp.shape[0],1))+0.00001

# Define the prior and the likelihood according to pyro syntax
mu_prior = 0.0
sigma_prior = 10
mean_likelihood = torch.zeros((n_samples,1))
sigma_likelihood = torch.ones((n_samples,1))*0.5182385

def model_noise(x_observed, u_observed):
    # Prior is a gaussian distriubtion with mean 0 and standard deviation 0.1
    #print("prior:", mu_prior, sigma_prior)
    prior_w = dist.Normal(mu_prior, sigma_prior)
    # Sample from the prior
    w = pyro.sample("w", prior_w)
    w = w.expand(x_observed.shape[0], 1)
    #print(x_observed.transpose(1,0),w.transpose(1,0))
    #print("w:",w)
    # Data likelihood is a gaussian distriubtion with mean G(x,w)=wx and standard deviation 0.1
    inputs_sigma = torch.cat([x_observed, sigma_likelihood], 1)
    inputs = torch.cat([x_observed, w], 1)
    #mean_likelihood = G(inputs)
    #sigma_likelihood = torch.from_numpy(np.arange(0.1,1.1,1/n_samples))
    mean_likelihood = G(inputs)
    #print("sigma:",sigma_likelihood)
    # Data likelihood is a gaussian distriubtion with mean G(x,w)=wx and standard deviation 0.1
    #sigma_likelihood = torch.ones((n_samples,1))*0.1
    #print(sigma_likelihood, mean_likelihood)
    #sigma_likelihood = G(inputs)
    #print(mean_likelihood.shape,sigma_likelihood.shape)
    #print(mean_likelihood,sigma_likelihood)
    sigma_t = G_sigma(inputs_sigma)
    likelihood = dist.Normal(mean_likelihood, sigma_t)
    # Sample from the likelihood
    u_sampled = pyro.sample("obs", likelihood, obs=u_observed)
    #print("u:",u_sampled)
    #print(u_sampled)

n_samples_nuts = 1000
nuts_kernel = NUTS(model_noise)
posterior = MCMC(nuts_kernel, num_samples=n_samples_nuts, warmup_steps=1000, initial_params={"w": torch.tensor(mu_prior)})
posterior.run(t,measurement_at_t)
hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
print(hmc_samples["w"].mean())
w = hmc_samples["w"].mean()*torch.ones((t.shape[0],1))
#print(w, t.shape)

plt.figure()
inputs = torch.cat([t, w], 1)
plt.plot(t,model_best(inputs).detach().numpy())
inputs = torch.cat([t, w], 1)
t_meas = torch.from_numpy(data[:, 0].transpose()).reshape((n_samples,1)).float()
measurement_at_t = torch.from_numpy(data[:, 1].transpose()).reshape((n_samples,1)).float()
plt.scatter(t_meas,measurement_at_t, alpha=0.7)
plt.plot(t,best_model(inputs).detach().numpy())
plt.show()()

plt.figure(dpi=150)
sns.distplot(hmc_samples["w"], label="Approximate", norm_hist=True)
plt.xlabel("w")
plt.title("Posterior Distrubtion")
plt.legend()
plt.show()

fname = os.path.join(dirstr+"/Task4b.txt")
np.savetxt(fname, hmc_samples["w"].mean(),delimiter=" ")
