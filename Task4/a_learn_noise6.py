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

def G(coeff_mean, coeff_sigma, x):
    # Corresponds to G(x, w) from the notes, exact underlying model
    # this might come from the solution of PDE, for example
    x_mean = x - coeff_mean[:,4]
    x_sigma = x - coeff_sigma[:,4]
    mean = coeff_mean[:,0] + coeff_mean[:,1]*x_mean + coeff_mean[:,2]*torch.pow(x_mean,2) + coeff_mean[:,3]*torch.pow(x_mean,3)
    sigma = torch.abs(coeff_sigma[:,0] + coeff_sigma[:,1]*x_sigma + coeff_sigma[:,2]*torch.pow(x_sigma,2) + coeff_sigma[:,3]*torch.pow(x_sigma,3))
    return mean, sigma

# Define the prior and the likelihood according to pyro syntax
mu_prior = torch.ones((10,1))*2
sigma_prior = torch.ones((10,1))*10
mean_likelihood = torch.zeros((n_samples,1))
#sigma_likelihood = 0.1

def model_noise(x_observed, u_observed):
    # Prior is a gaussian distriubtion with mean 0 and standard deviation 0.1
    #print("prior:", mu_prior, sigma_prior)
    prior_w = dist.Normal(mu_prior, sigma_prior)
    # Sample from the prior
    w = pyro.sample("w", prior_w)
#    print(w)
    w = w.reshape(1,10)
    print(w)
    #print(w)
    w = w.expand(x_observed.shape[0], 10)
    #print(w)
    #print("w:",w)
    # Data likelihood is a gaussian distriubtion with mean G(x,w)=wx and standard deviation 0.1
    inputs = torch.cat([x_observed, w], 1)
    #mean_likelihood = G(inputs)
    #sigma_likelihood = torch.from_numpy(np.arange(0.1,1.1,1/n_samples))
    mean_likelihood, sigma_likelihood = G(w[:,0:5],w[:,5:10],x_observed)
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

n_samples_nuts = 50
nuts_kernel = NUTS(model_noise)
posterior = MCMC(nuts_kernel, num_samples=n_samples_nuts, warmup_steps=50, initial_params={"w": torch.tensor(mu_prior)})
posterior.run(t,measurement_at_t)
hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
print(hmc_samples["w"])
mean_coeff = torch.zeros(1,5)
sigma_coeff = torch.zeros(1,5)
mean_coeff[0,0] = torch.tensor([hmc_samples["w"][:,0].mean()])
mean_coeff[0,1] = torch.tensor([hmc_samples["w"][:,1].mean()])
mean_coeff[0,2] = torch.tensor([hmc_samples["w"][:,2].mean()])
mean_coeff[0,3] = torch.tensor([hmc_samples["w"][:,3].mean()])
mean_coeff[0,4] = torch.tensor([hmc_samples["w"][:,4].mean()])
sigma_coeff[0,0] = torch.tensor([hmc_samples["w"][:,5].mean()])
sigma_coeff[0,1] = torch.tensor([hmc_samples["w"][:,6].mean()])
sigma_coeff[0,2] = torch.tensor([hmc_samples["w"][:,7].mean()])
sigma_coeff[0,3] = torch.tensor([hmc_samples["w"][:,8].mean()])
sigma_coeff[0,4] = torch.tensor([hmc_samples["w"][:,9].mean()])
mean, sigma = G(mean_coeff, sigma_coeff, t)
plt.plot(t,mean)
plt.show

plt.figure(dpi=150)
sns.distplot(hmc_samples["w"][1], label="Approximate", norm_hist=True)
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
