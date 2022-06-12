import numpy as np
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time
from datetime import datetime

torch.set_num_threads(8)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
os.mkdir("run_" + dt_string)


torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_):
        self.alpha_f = 0.05
        self.alpha_s = 0.08
        self.h_f = 5
        self.h_s = 8
        self.T_hot = 4
        self.T_0 = 1
        self.U_f = 1

        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[0, 1],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=2,
                                              n_hidden_layers=8,
                                              neurons=40,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial condition to solve the heat equation u0(x)=-sin(pi x)
    def initial_condition(self, x):
        return torch.ones((2*x.shape[0],1))*self.T_0

    # Exact solution for the heat equation ut = u_xx with the IC above
    def exact_solution(self, inputs):
        t = inputs[:, 0]
        x = inputs[:, 1]

        u = -torch.exp(-np.pi ** 2 * t) * torch.sin(np.pi * x)
        return u

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)

        return torch.cat([input_tb[:,0].reshape(input_tb.shape[0],1),input_tb[:,1].reshape(input_tb.shape[0],1)], 0), output_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        T_0_tens = torch.ones((input_sb.shape[0],1))*self.T_0
        T_hot_tens = torch.ones((input_sb.shape[0],1))*self.T_hot
        denom_0 = torch.ones((input_sb.shape[0],1)) + torch.exp(-200*(input_sb_0[:,0].reshape(input_sb_0.shape[0],1)-0.25*torch.ones((input_sb_0.shape[0],1))))

        output_sbf_0 = (T_hot_tens-T_0_tens)/denom_0 + T_0_tens
        output_sbf_L = torch.zeros((input_sb.shape[0], 1))
        output_sbs_0 = torch.zeros((input_sb.shape[0], 1))
        output_sbs_L = torch.zeros((input_sb.shape[0], 1))

        input_sb_0_t = input_sb_0[:,0].reshape((n_sb,1))
        input_sb_0_x = input_sb_0[:,1].reshape((n_sb,1))
        input_sb_L_t = input_sb_L[:,0].reshape((n_sb,1))
        input_sb_L_x = input_sb_L[:,1].reshape((n_sb,1))

        return torch.cat([input_sb_0_t, input_sb_L_t, input_sb_0_x,
            input_sb_L_x], 0), torch.cat([output_sbf_0, output_sbf_L, output_sbs_0, output_sbs_L,], 0)
    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=4*self.space_dimensions*self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb,
            output_tb), batch_size=2*self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        input_tb_t = input_tb[0:n_tb].reshape(n_tb,1)
        input_tb_x = input_tb[n_tb:2*n_tb].reshape(n_tb,1)
        input_tb_stacked = torch.cat([input_tb_t,input_tb_x],1)
        u_pred_tb = self.approximate_solution(input_tb_stacked)
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        input_sb_0_t = input_sb[0:n_sb].reshape(n_sb,1)
        input_sb_L_t = input_sb[n_sb:2*n_sb].reshape(n_sb,1)
        input_sb_0_x = input_sb[2*n_sb:3*n_sb].reshape(n_sb,1)
        input_sb_L_x = input_sb[3*n_sb:4*n_sb].reshape(n_sb,1)
        input_sb_0_stacked = torch.cat([input_sb_0_t,input_sb_0_x],1)
        input_sb_L_stacked = torch.cat([input_sb_L_t,input_sb_L_x],1)
        u_pred_sb = self.approximate_solution(torch.cat([input_sb_0_stacked,
            input_sb_L_stacked],0))

        return u_pred_sb

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        u_f =u[:,0]
        u_s =u[:,1]
        grad_u_f = torch.autograd.grad(u[:,0].sum(), input_int, create_graph=True)[0]
        grad_u_f_t = grad_u_f[:, 0]
        grad_u_f_x = grad_u_f[:, 1]
        grad_u_f_xx = torch.autograd.grad(grad_u_f_x.sum(), input_int, create_graph=True)[0][:, 1]
        grad_u_s = torch.autograd.grad(u[:,1].sum(), input_int, create_graph=True)[0]
        grad_u_s_t = grad_u_s[:, 0]
        grad_u_s_x = grad_u_s[:, 1]
        grad_u_s_xx = torch.autograd.grad(grad_u_s_x.sum(), input_int, create_graph=True)[0][:, 1]

        residual_1 = grad_u_f_t + self.U_f*grad_u_f_x + self.h_f*(u_f-u_s) - self.alpha_f*grad_u_f_xx
        residual_2 = grad_u_s_t - self.h_s*(u_f-u_s) - self.alpha_s*grad_u_s_xx
        return torch.cat([residual_1,residual_2],0).reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        inp_train_sb.requires_grad = True
        u_pred_tb = self.apply_initial_condition(inp_train_tb)
        u_pred_tbf = u_pred_tb[:,0].reshape(u_pred_tb.shape[0],1)
        u_pred_tbs = u_pred_tb[:,1].reshape(u_pred_tb.shape[0],1)
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_sbf = u_pred_sb[:,0].reshape(u_pred_sb.shape[0],1)
        u_pred_sbs = u_pred_sb[:,1].reshape(u_pred_sb.shape[0],1)
        u_pred_sbf_grad = torch.autograd.grad(u_pred_sbf.sum(),inp_train_sb, create_graph=True)[0]
        u_pred_sbf_gradx = u_pred_sbf_grad[2*n_sb:4*n_sb].reshape((2*n_sb,1))
        u_pred_sbs_grad = torch.autograd.grad(u_pred_sbs.sum(),inp_train_sb, create_graph=True)[0]
        u_pred_sbs_gradx =  u_pred_sbs_grad[2*n_sb:4*n_sb].reshape((2*n_sb,1))



        u_pred_sb_seq = torch.cat([u_pred_sbf[0:n_sb],
            u_pred_sbf_gradx[n_sb:2*n_sb], u_pred_sbs_gradx[0:n_sb],
                u_pred_sbs_gradx[n_sb:2*n_sb]],0)
        assert (u_pred_sb_seq.shape[1] == u_train_sb.shape[1])
        u_pred_tb_seq = torch.cat([u_pred_tbf,u_pred_tbs],0)
        assert (u_pred_tb_seq.shape[1] == u_train_tb.shape[1])
        r_int = self.compute_pde_residual(inp_train_int)
        r_sb = u_train_sb - u_pred_sb_seq
        r_tb = u_train_tb - u_pred_tb_seq

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)

        loss_u = loss_sb + loss_tb

        loss = torch.log10(self.lambda_u * (loss_sb + loss_tb) + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_u).item(), 4), "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)
        fname = os.path.join("run_"+dt_string+"/config.txt")
        with open(fname, "a") as config:
            floss = float(history[-1])
            config.write("Final Loss: " + str(floss))

        print('Final Loss: ', history[-1])

        return history

    ################################################################################################
    def plotting(self):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output_f = self.approximate_solution(inputs)[:,0].reshape(-1, )
        output_s = self.approximate_solution(inputs)[:,1].reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(),
                c=output_f.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        axs[0].set_title("fluid_samples")
        im1 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(),
                c=output_s.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im1, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[1].set_title("solid_samples")
        plt.show()

        fname = os.path.join("TestingData.txt")
        output_dimension = 2
        inputs = torch.from_numpy(np.loadtxt(fname,
            delimiter=",",skiprows=1)).float()

        output_f = self.approximate_solution(inputs)[:,0].reshape(-1, )
        output_s = self.approximate_solution(inputs)[:,1].reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(),
                c=output_f.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        axs[0].set_title("fluid_test")
        im1 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(),
                c=output_s.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im1, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[1].set_title("solid_test")
        plt.show()

n_int = 512
n_sb = 128
n_tb = 128

pinn = Pinns(n_int, n_sb, n_tb)

n_epochs = 1
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1)# * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(), lr=float(0.005))
# Plot the input training points
#input_sb_, output_sb_ = pinn.add_spatial_boundary_points()
#input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
#input_int_, output_int_ = pinn.add_interior_points()
#input_sb_t = input_sb_[0:2*n_sb]
#input_sb_x = input_sb_[2*n_sb:4*n_sb]
#output_sb_f = output_sb_[0:2*n_sb]
#output_sb_s = output_sb_[2*n_sb:4*n_sb]
#input_tb_t = input_tb_[0:n_sb]
#input_tb_x = input_tb_[n_sb:2*n_sb]
#output_tb_f = output_sb_[0:n_sb]
#output_tb_s = output_sb_[n_sb:2*n_sb]
#input_sb_stacked = torch.cat([input_sb_t,input_sb_x],1)
#ouput_sb_stacked = torch.cat([output_sb_f,output_sb_s],1)
#input_tb_stacked = torch.cat([input_tb_t,input_tb_x],1)
#output_tb_stacked = torch.cat([output_tb_f,output_tb_s],1)

#plt.figure(figsize=(16, 8), dpi=150)
#plt.scatter(input_sb_stacked[:, 1].detach().numpy(), input_sb_stacked[:, 0].detach().numpy(), label="Boundary Points")
#plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
#plt.scatter(input_tb_stacked[:, 1].detach().numpy(), input_tb_stacked[:, 0].detach().numpy(), label="Initial Points")
#plt.xlabel("x")
#plt.ylabel("t")
#plt.legend()
#plt.show()

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)
pinn.plotting()

torch.save(pinn.approximate_solution.state_dict(), "run_"+dt_string+"/model.pth")

fname = os.path.join("TestingData.txt")
output_dimension = 2
test_data = np.loadtxt(fname, delimiter=",",skiprows=1)
prediction = np.ndarray((test_data.shape[0], test_data.shape[1]+output_dimension))
input_dim = test_data.shape[1]
prediction[:,0:input_dim] = test_data
prediction = torch.from_numpy(prediction)
prediction[:,input_dim:input_dim+output_dimension] = pinn.approximate_solution(prediction[:,0:input_dim].float())
prediction = prediction.detach().numpy()
fname = os.path.join("run_" + dt_string+"/SubTask6.txt")
np.savetxt(fname, prediction[:,input_dim], header="t,x,tf,ts")
