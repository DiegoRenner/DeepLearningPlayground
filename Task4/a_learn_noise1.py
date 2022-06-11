import pyro
import pyro.distributions as dist
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

std_opt = torch.tensor(torch.ones(n_samples,1)*0.5,requires_grad=True)
#dist_opt.requires_grad = True
mean = model(t)

##model = torch.load("models/model.pth")
#v_opt = torch.tensor([0.5] , requires_grad=True)
#
#v_min = 50
#v_max = 400
#D_min = 2
#D_max = 20
#v_min_mod = 0
#v_max_mod = 1
#D_min_mod = 0
#D_max_mod = 1
#
#D_grid = np.arange(0,1,0.001)
#
optimizer = optim.LBFGS([std_opt], lr=float(0.00001), max_iter=50000,
        max_eval=50000, history_size=100, line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
optimizer.zero_grad()
cost = list([0])
def closure():
    #G = dist.Normal(mean, std_opt)
    G = torch.sum(torch.log(2*np.pi*std_opt))# + torch.sum(torch.pow((measurement_at_t-mean)/std_opt,2))
    #G = 2*np.pi*torch.sum(std_opt) #+torch.sum(torch.pow((measurement_at_t-mean)/std_opt,2))/2
    #G = torch.abs(model(torch.clamp(v_opt, min=v_min_mod, max=v_max_mod))-torch.tensor([0.45]))
    cost[0] = G
    G.backward()
    return G
optimizer.step(closure=closure)
print("Minimizer: ", std_opt)
#inputs = torch.cat((torch.tensor([D]).float(),torch.clamp(v_opt, min=v_min_mod, max=v_max_mod)))
#print("Corresponding flux values: ", model(inputs))
