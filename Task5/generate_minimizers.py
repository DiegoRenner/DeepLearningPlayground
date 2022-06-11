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
model = NeuralNet(2, 1, 2, 5, 0.0, 2, "Tanh")
model.load_state_dict(torch.load("models/model.pth"))
model.eval()
model.requires_grad_ = True

#model = torch.load("models/model.pth")
v_opt = torch.tensor([0.5] , requires_grad=True)

v_min = 50
v_max = 400
D_min = 2
D_max = 20
v_min_mod = 0
v_max_mod = 1
D_min_mod = 0
D_max_mod = 1

D_grid = np.arange(0,1,0.001)

for D in D_grid:
    optimizer = optim.LBFGS([v_opt], lr=float(0.00001), max_iter=500000,
            max_eval=500000, history_size=100, line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
    optimizer.zero_grad()
    cost = list([0])
    def closure():
        inputs = torch.cat((torch.tensor([D],requires_grad=False).float(),torch.clamp(v_opt, min=v_min_mod, max=v_max_mod)))
        G = torch.abs(model(inputs)-torch.tensor([0.45]))
        #G = torch.abs(model(torch.clamp(v_opt, min=v_min_mod, max=v_max_mod))-torch.tensor([0.45]))
        cost[0] = G
        G.backward()
        return G
    optimizer.step(closure=closure)
    print("Minimizer: ", torch.clamp(v_opt, min=v_min_mod, max=v_max_mod))
    inputs = torch.cat((torch.tensor([D]).float(),torch.clamp(v_opt, min=v_min_mod, max=v_max_mod)))
    print("Corresponding flux values: ", model(inputs))

#min_inputs = 0
#max_inputs = 1
#y_opt = torch.tensor(torch.tensor([0.5, 0.5]), requires_grad=True)
#y_init = torch.clone(y_opt)
#
#optimizer = optim.LBFGS([y_opt], lr=float(0.00001), max_iter=50000, max_eval=50000, history_size=100, line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
#
#optimizer.zero_grad()
#cost = list([0])
#
#
#def closure():
#    G = (torch.clamp(y_opt, min=min_inputs, max=max_inputs)-min_inputs)/(max_inputs - min_inputs)
#    print(G)
#    cost[0] = G
#    G.backward()
#    return G
#
#
#optimizer.step(closure=closure)
#print("Minimizer: ", torch.clamp(y_opt, min=min_inputs, max=max_inputs))
#print("Corresponding flux values: ", model((torch.clamp(y_opt, min=min_inputs, max=max_inputs)-min_inputs)/(max_inputs - min_inputs)))
