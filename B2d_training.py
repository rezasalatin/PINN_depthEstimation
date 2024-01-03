"""
PINN training for Boussinesq 2d
@author: Reza Salatin
December 2023
w/ Pytorch
"""

import torch
from torch import nn
from collections import OrderedDict
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
import os
from physics import Boussinesq_simple as physics

np.random.seed(1234)

# CUDA support
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    device = torch.device('cuda')
    print("Device in use: GPU")
else:
    torch.manual_seed(1234)
    device = torch.device('cpu')
    print("Device in use: CPU")

# Get current date and time
now = datetime.datetime.now()
folder_name = now.strftime("log_%Y%m%d_%H%M")
# Create a directory with the folder_name
log_dir = f"../log/{folder_name}"
os.makedirs(log_dir, exist_ok=True)
    
# the deep neural network
class DNN(nn.Module):

    def __init__(self, layers):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(self._build_layers(layers))

    def _build_layers(self, layers):
        layer_list = []
        num_layers = len(layers)

        for i in range(num_layers - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            DNN._initialize_layer(linear_layer, zero_bias=(i < num_layers - 2))

            layer_list.append((f'layer_{i}', linear_layer))
            if i < num_layers - 2:
                layer_list.append((f'activation_{i}', nn.Tanh()))

        return OrderedDict(layer_list)

    def _initialize_layer(layer, zero_bias=True):
        """Initialize a layer with Xavier uniform weights and optional zeroed bias."""
        nn.init.xavier_uniform_(layer.weight)
        if zero_bias:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass of the network."""
        return self.layers(x)
    

# the physics-guided neural network
class PINN():

    # Initializer
    def __init__(self, X_u, X_f, layers, X_min, X_max, AdamIt, LBFGSIt):
        
        # temporal and spatial information for observations
        self.t_u = torch.tensor(X_u[:, 0:1]).float().to(device)
        self.x_u = torch.tensor(X_u[:, 1:2]).float().to(device)
        self.y_u = torch.tensor(X_u[:, 2:3]).float().to(device)

        self.h_u = torch.tensor(X_u[:, 3:4]).float().to(device)
        self.z_u = torch.tensor(X_u[:, 4:5]).float().to(device)
        self.u_u = torch.tensor(X_u[:, 5:6]).float().to(device)
        self.v_u = torch.tensor(X_u[:, 6:7]).float().to(device)
        
        # temporal and spatial information for physics
        self.t_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_f[:, 2:3], requires_grad=True).float().to(device)                
        self.z_f = torch.tensor(X_f[:, 3:4]).float().to(device)

        # max and min values for normalization
        self.vals_min = torch.tensor(X_min).float().to(device)
        self.vals_max = torch.tensor(X_max).float().to(device)
        
        # layers of NN
        self.layers = layers
        self.dnn = DNN(layers).to(device)

        # Adam and LBFGS max iterations
        self.AdamIt = AdamIt
        self.LBFGSIt = LBFGSIt

        # Initialize iteration counter
        self.iter = 0
        

    # Initialize optimizers
    def init_optimizers(self):

        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr=1e-4 # learning rate
        )

        # Define learning rate scheduler for Adam
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size=10000, 
            gamma=0.8
        )

        # L-BFGS optimizer
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=self.LBFGSIt, 
            max_eval=self.LBFGSIt*1.25, 
            history_size=100,
            tolerance_grad=1e-5, 
            tolerance_change=1e-7,
            line_search_fn="strong_wolfe"
        )


    # Normalize and feed inputs to NN to get predictions
    def net_u(self, t, x, y, z):

        # input normalization between [-1, 1]
        t = 2.0 * (t - self.vals_min[0])/(self.vals_max[0]-self.vals_min[0]) - 1.0
        x = 2.0 * (x - self.vals_min[1])/(self.vals_max[1]-self.vals_min[1]) - 1.0
        y = 2.0 * (y - self.vals_min[2])/(self.vals_max[2]-self.vals_min[2]) - 1.0
        z = 2.0 * (z - self.vals_min[4])/(self.vals_max[4]-self.vals_min[4]) - 1.0
        output = self.dnn(torch.cat([t, x, y, z], dim=1))

        return output
    

    # Loss function
    def loss_func(self):
        
        output_u_pred = self.net_u(self.t_u, self.x_u, self.y_u, self.z_u)
        
        h_pred = output_u_pred[:, 0:1].to(device)
        z_pred = output_u_pred[:, 1:2].to(device)
        u_pred = output_u_pred[:, 2:3].to(device)
        v_pred = output_u_pred[:, 3:4].to(device)
        
        loss_comp_h = torch.mean((self.h_u - h_pred)**2)
        loss_comp_z = torch.mean((self.z_u - z_pred)**2)
        loss_comp_u = torch.mean((self.u_u - u_pred)**2)
        loss_comp_v = torch.mean((self.v_u - v_pred)**2)

        # Fidelity loss
        weight_h, weight_z, weight_u, weight_v = 1.0, 1.0, 1.0, 1.0
        loss_u = weight_h * loss_comp_h + weight_z * loss_comp_z \
            + weight_u * loss_comp_u + weight_v * loss_comp_v
        
        # Residual loss
        output_f_pred = self.net_u(self.t_f, self.x_f, self.y_f, self.z_f)
        #loss_f = physics(output_f_pred, self.t_f, self.x_f, self.y_f, device)
        
        # Total loss
        weight_fidelity, weight_residuals = 1.0, 0.0
        loss = weight_fidelity * loss_u
                
        # iteration (epoch) counter
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % 
                (self.iter, loss_u.item(), loss_u.item(), loss.item()))

        return loss

    
    # Model training with two optimizers
    def train(self):

        self.dnn.train()

        # Training with Adam optimizer
        for i in range(self.AdamIt):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_Adam.step()
            self.scheduler_Adam.step()

        # Training with L-BFGS optimizer
        def closure():
            self.optimizer_LBFGS.zero_grad()  # Zero gradients for LBFGS optimizer
            loss = self.loss_func()
            loss.backward()
            return loss
        self.optimizer_LBFGS.step(closure)

    
    # Testing: Prediction
    def predict(self, X):

        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        z = torch.tensor(X[:, 3:4]).float().to(device)

        output_pred = self.net_u(t, x, y, z)
        
        h_pred = output_pred[:, 0:1].to(device)
        z_pred = output_pred[:, 1:2].to(device)
        u_pred = output_pred[:, 2:3].to(device)
        v_pred = output_pred[:, 3:4].to(device)

        return h_pred, z_pred, u_pred, v_pred
    

    # Testing: On the fly residual loss calculator
    def loss_func_otf(self, new_data):

        new_t, new_x, new_y, new_h, new_z, new_u, new_v = [torch.tensor(new_data[:, i:i+1]).float().to(device) for i in range(3, 7)]  # Adjust indices based on new_data structure
        output_f_pred = torch.cat([new_h, new_z, new_u, new_v], dim=1)
        loss = physics(output_f_pred, new_t, new_x, new_y, device)

        return loss
    
    # Testing: On the fly model update with L-BFGS
    def update_model_otf(self, new_data):

        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss_func_otf(new_data)
            loss.backward()
            return loss
        
        # Perform one optimization step
        self.optimizer_LBFGS.step(closure)
    

# Main
if __name__ == "__main__": 
       
    # Define training points + iterations for Adam and LBFGS
    Ntrain = 4800
    AdamIt = 0
    LBFGSIt = 50000
    
    # Define input, hidden, and output layers
    input_features = 4 # t, x, y, eta
    hidden_layers = 20
    hidden_width = 20
    output_features = 4 # h, eta, u, v
    layers = [input_features] + [hidden_width] * hidden_layers + [output_features]
    
    funwave_dir = '../data/output_reg_pinn'
    x_grid = 251
    y_grid = 501
    dx = 2
    dy = 2
    
    training_data_dir = '../data/beach2d_reg.csv'
    
    # Extract all data from csv file.
    data = np.genfromtxt(training_data_dir, delimiter=' ').astype(np.float32) # load data
    t_all = data[:, 0:1].astype(np.float64)
    x_all = data[:, 1:2].astype(np.float64)
    y_all = data[:, 2:3].astype(np.float64)
    h_all = data[:, 3:4].astype(np.float64)
    z_all = data[:, 4:5].astype(np.float64)
    u_all = data[:, 5:6].astype(np.float64)
    v_all = data[:, 6:7].astype(np.float64)
    
    X_star = np.hstack((t_all, x_all, y_all, h_all, z_all, u_all, v_all))
    # get data range for data normalization for training only
    X_star_min = np.min(X_star, axis=0)
    X_star_max = np.max(X_star, axis=0)
    
    # select training points randomly from the domain.
    idx = np.random.choice(X_star.shape[0], Ntrain, replace=False)
    X_u_star = X_star[idx,:]
    
    # select all available points (gauges) for residuals
    dx_interval = dx*10
    dy_interval = dy*10
    X_f_star = np.empty((0, 4))  # Assuming you have 4 columns: T, X, Y, Z
    x_f = np.arange(0, (x_grid-1)*dx + 1, dx).astype(np.float64)
    y_f = np.arange(0, (y_grid-1)*dy + 1, dy).astype(np.float64)
    X_f, Y_f = np.meshgrid(x_f, y_f)
    X_f = X_f[::dx_interval, ::dy_interval]
    Y_f = Y_f[::dx_interval, ::dy_interval]
    # Flatten the X and Y arrays
    X_f_flat = X_f.flatten().reshape(-1, 1)
    Y_f_flat = Y_f.flatten().reshape(-1, 1)
    
    for t in  [200, 211, 223, 233, 250, 253, 267, 277, 289, 300]:
        
        file_suffix = str(t).zfill(5)  # Pad the number with zeros to make it 5 digits
        
        T_f = np.full((x_grid, y_grid), t, dtype=np.float64)
        T_f = T_f[::dx_interval, ::dy_interval]
        T_f_flat = T_f.flatten().reshape(-1, 1)
        
        Z_f = np.loadtxt(funwave_dir + f'/eta_{file_suffix}')
        Z_f = Z_f[::dx_interval, ::dy_interval]
        Z_f_flat = Z_f.flatten().reshape(-1, 1)

        # Create new data for this iteration
        new_data = np.hstack((T_f_flat, X_f_flat, Y_f_flat, Z_f_flat))
        # Append new data to X_f_star
        X_f_star = np.vstack((X_f_star, new_data))

    # set up the pinn model
    model = PINN(X_u_star, X_f_star, layers, X_star_min, X_star_max, AdamIt, LBFGSIt)
    model.init_optimizers()  # Initialize optimizers
    
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn.state_dict(), os.path.join(log_dir, 'model.ckpt')) # Only the parameters
    torch.save(model.dnn, os.path.join(log_dir, 'complete_model.pth'))      # Entire model

    ###### Testing

    t_final = 500

    x_test = np.arange(0, (x_grid-1)*dx + 1, dx).astype(np.float64)
    y_test = np.arange(0, (y_grid-1)*dy + 1, dy).astype(np.float64)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    # Flatten the X and Y arrays
    X_test_flat = X_test.flatten().reshape(-1, 1)
    Y_test_flat = Y_test.flatten().reshape(-1, 1)

    # get bathymetry file and flatten it
    h_test = np.loadtxt(funwave_dir + '/dep.out')
    h_test_flat = h_test.flatten().reshape(-1, 1)

    for t in range(0,t_final+1):

        T_test = np.full((x_grid, y_grid), t, dtype=np.float64)
        T_test_flat = T_test.flatten().reshape(-1, 1)

        file_suffix = str(t).zfill(5)  # Pad the number with zeros to make it 5 digits
        Z_test = np.loadtxt(funwave_dir + f'/eta_{file_suffix}')  # Construct the file name
        Z_test_flat = Z_test.flatten().reshape(-1, 1)
        
        # make inputs ready for NN
        X_star = np.hstack((T_test_flat, X_test_flat, Y_test_flat, Z_test_flat))

        # feed into NN and get outpus
        h_pred, z_pred, u_pred, v_pred = model.predict(X_star)
        
        h_pred = h_pred.detach().cpu().numpy()
        z_pred = z_pred.detach().cpu().numpy()
        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()        

        # Reshape predictions to match original grid shape
        h_pred_reshaped = h_pred.reshape(X_test.shape)
        z_pred_reshaped = z_pred.reshape(X_test.shape)
        u_pred_reshaped = u_pred.reshape(X_test.shape)
        v_pred_reshaped = v_pred.reshape(X_test.shape)

        ########## Model update during testing
        # Format the predictions to match the training data structure
        new_data = np.hstack((T_test_flat, X_test_flat, Y_test_flat, h_pred, z_pred, u_pred, v_pred))
        # Update the model using new predictions
        model.update_model_otf(new_data)
        
        # for all figures
        fsize = 14
        x_limits = [150, 500]
        y_limits = [0, 1000]

        if t % 6 == 0:

            ########## Fig 1
            # Plotting figure 1 for eta
            fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
            # X, Y, Z plot
            cmap1 = ax.pcolor(X_test, Y_test, Z_test, shading='auto')
            cbar1 = fig.colorbar(cmap1, ax=ax)
            cbar1.set_label('eta_{real} (m)')
            ax.set_xlabel('X (m)', fontsize=fsize)
            ax.set_ylabel('Y (m)', fontsize=fsize)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            # Save the plot with file number in the filename
            plt.savefig(f'../plots/plots/Eta_true_{file_suffix}.png', dpi=300)
            plt.tight_layout()
            plt.show()
            plt.close()

            ########## Fig 2
             # Plotting figure 1 for eta
            fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
            cmap1 = ax.pcolor(X_test, Y_test, h_pred_reshaped, shading='auto')
            cbar1 = fig.colorbar(cmap1, ax=ax)
            cbar1.set_label('bathymetry (m)')
            ax.set_xlabel('X (m)', fontsize=fsize)
            ax.set_ylabel('Y (m)', fontsize=fsize)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            # Save the plot with file number in the filename
            plt.savefig(f'../plots/plots/h_pred_{file_suffix}.png', dpi=300)
            plt.tight_layout()
            plt.show()
            plt.close()

            # Print the current value of t
            print(f'Figures for t = {t} are plotted!')
        
