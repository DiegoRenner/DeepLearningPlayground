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
from Common_mod import NeuralNet, fit, run_configuration
import seaborn as sb

model_dict = torch.load("models/model.pth")
model = NeuralNet(1, 1, 4, 20, 0.0004, 2, "Tanh")
model.load_state_dict(torch.load("models/model.pth"))
model.eval()
model.requires_grad_ = True

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
    x = inp[:, 0]
    w = inp[:, 1]
    g = torch.abs(x*w) + 0.1
    return g

# Define the prior and the likelihood according to pyro syntax
mu_prior = 0.0
sigma_prior = 1
mean_likelihood = torch.zeros((n_samples,1))
#sigma_likelihood = 0.1

def model_noise(x_observed, u_observed):
    # Prior is a gaussian distriubtion with mean 0 and standard deviation 0.1
    #print("prior:", mu_prior, sigma_prior)
    prior_w = dist.Normal(mu_prior, sigma_prior)
    # Sample from the prior
    w = pyro.sample("w", prior_w)
    print(w)
    w = w.expand(x_observed.shape[0], 1)
    print(w)
    #print("w:",w)
    # Data likelihood is a gaussian distriubtion with mean G(x,w)=wx and standard deviation 0.1
    inputs = torch.cat([x_observed, w], 1)
    #mean_likelihood = G(inputs)
    #sigma_likelihood = torch.from_numpy(np.arange(0.1,1.1,1/n_samples))
    mean_likelihood = 0.0
    sigma_likelihood = G(inputs)
    #print("mean:",mean_likelihood)
    #print("sigma:",sigma_likelihood)
    # Data likelihood is a gaussian distriubtion with mean G(x,w)=wx and standard deviation 0.1
    #sigma_likelihood = torch.ones((n_samples,1))*0.1
    #print(sigma_likelihood, mean_likelihood)
    #sigma_likelihood = G(inputs)
    #print(mean_likelihood.shape,sigma_likelihood.shape)
    #print(mean_likelihood,sigma_likelihood)
    likelihood = dist.Normal(mean_likelihood, sigma_likelihood)
    # Sample from the likelihood
    u_sampled = pyro.sample("obs", likelihood, obs=u_observed)
    #print("u:",u_sampled)
    #print(u_sampled)

n_samples_nuts = 10000
nuts_kernel = NUTS(model_noise)
mu_prior = 0.2
posterior = MCMC(nuts_kernel, num_samples=n_samples_nuts, warmup_steps=1000, initial_params={"w": torch.tensor(mu_prior)})
posterior.run(t,measurement_at_t-model(t))
hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}

plt.figure(dpi=150)
sns.distplot(hmc_samples["w"], label="Approximate", norm_hist=True)
plt.xlabel("w")
plt.title("Posterior Distrubtion")
plt.legend()
plt.show()

#std_opt = torch.tensor(torch.ones(n_samples,1)*0.5,requires_grad=True)
##dist_opt.requires_grad = True
#mean = model(t)
#
###model = torch.load("models/model.pth")
##v_opt = torch.tensor([0.5] , requires_grad=True)
##
##v_min = 50
##v_max = 400
##D_min = 2
##D_max = 20
##v_min_mod = 0
##v_max_mod = 1
##D_min_mod = 0
##D_max_mod = 1
##
##D_grid = np.arange(0,1,0.001)
##
#optimizer = optim.LBFGS([std_opt], lr=float(0.00001), max_iter=50000,
#        max_eval=50000, history_size=100, line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
#optimizer.zero_grad()
#cost = list([0])
#def closure():
#    #G = dist.Normal(mean, std_opt)
#    G = torch.sum(torch.log(2*np.pi*std_opt))# + torch.sum(torch.pow((measurement_at_t-mean)/std_opt,2))
#    #G = 2*np.pi*torch.sum(std_opt) #+torch.sum(torch.pow((measurement_at_t-mean)/std_opt,2))/2
#    #G = torch.abs(model(torch.clamp(v_opt, min=v_min_mod, max=v_max_mod))-torch.tensor([0.45]))
#    cost[0] = G
#    G.backward()
#    return G
#optimizer.step(closure=closure)
#print("Minimizer: ", std_opt)
##inputs = torch.cat((torch.tensor([D]).float(),torch.clamp(v_opt, min=v_min_mod, max=v_max_mod)))
##print("Corresponding flux values: ", model(inputs))
