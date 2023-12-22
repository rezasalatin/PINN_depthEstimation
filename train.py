"""
PINN training for Boussinesq 2d
@author: Reza Salatin
@email: reza.salatin@whoi.edu
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
import json
from physics_functions import Boussinesq_simple as physics

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

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

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
        nn.init.xavier_uniform_(layer.weight)
        if zero_bias:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)
    
# the physics-guided neural network
class PINN():

    # Initializer
    def __init__(self, data_fid, data_res, layers, AdamIt, LBFGSIt):
        
        # temporal and spatial information for fidelity part
        num_columns = data_fid.shape[1]
        for i in range(num_columns):
            setattr(self, f'fid_in{i}', torch.tensor(data_fid[:, i:i+1]).float().to(device))
        
        # temporal and spatial information for physics part
        num_columns_res = data_res.shape[1]
        for i in range(num_columns_res):
            requires_grad = True if i < 3 else False  # Set requires_grad=True for the first three columns (t,x,y)
            setattr(self, f'res_in{i}', torch.tensor(data_res[:, i:i+1], requires_grad=requires_grad).float().to(device))

        # layers of NN
        self.layers = layers
        self.dnn = DNN(layers).to(device)

        # Initialize iteration counter
        self.iter = 0

    # Initialize optimizers
    def init_optimizers(self):

        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr = config['optimizer']['adam_learning_rate'] # learning rate
        )

        # Define learning rate scheduler for Adam
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size = config['optimizer']['scheduler_step_size'], 
            gamma = config['optimizer']['scheduler_gamma']
        )

        # L-BFGS optimizer
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr = config['optimizer']['lbfgs_learning_rate']
            max_iter = config['optimizer']['lbfgs_max_iteration'], 
            max_eval = config['optimizer']['lbfgs_max_evaluation'], 
            history_size = config['optimizer']['lbfgs_history_size'],
            tolerance_grad = config['optimizer']['lbfgs_tolerance_grad'],
            tolerance_change = config['optimizer']['lbfgs_tolerance_change'],
            line_search_fn = config['optimizer']['lbfgs_line_search_fn']
        )    

    # Loss function
    def loss_func(self):
        
        output_u_pred = self.dnn(self.t_u, self.x_u, self.y_u, self.z_u)
        
        h_pred = output_u_pred[:, 0:1].to(device)
        z_pred = output_u_pred[:, 1:2].to(device)
        u_pred = output_u_pred[:, 2:3].to(device)
        v_pred = output_u_pred[:, 3:4].to(device)
        
        loss_comp_h = torch.mean((self.h_u - h_pred)**2)
        loss_comp_z = torch.mean((self.z_u - z_pred)**2)
        loss_comp_u = torch.mean((self.u_u - u_pred)**2)
        loss_comp_v = torch.mean((self.v_u - v_pred)**2)

        # Fidelity loss
        # Extracting loss weights from the config
        weight_h = config['loss']['weight_h']
        weight_z = config['loss']['weight_z']
        weight_u = config['loss']['weight_u']
        weight_v = config['loss']['weight_v']
        loss_u = weight_h * loss_comp_h + weight_z * loss_comp_z \
            + weight_u * loss_comp_u + weight_v * loss_comp_v
        
        # Residual loss
        output_f_pred = self.dnn(self.t_f, self.x_f, self.y_f, self.z_f)
        loss_f = physics(output_f_pred, self.t_f, self.x_f, self.y_f, device)
        
        # Total loss
        weight_fidelity = config['loss']['weight_fidelity']
        weight_residual = config['loss']['weight_residual']
        loss = weight_fidelity * loss_u + weight_residual * loss_f
                
        # iteration (epoch) counter
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % 
                (self.iter, loss_u.item(), loss_f.item(), loss.item()))

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
        in0 = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        in1 = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        in2 = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        in3 = torch.tensor(X[:, 3:4]).float().to(device)
        output = self.dnn(in0, in1, in2, in3)
        out0 = output[:, 0:1].to(device)
        out1 = output[:, 1:2].to(device)
        out2 = output[:, 2:3].to(device)
        out3 = output[:, 3:4].to(device)
        return out0, out1, out2, out3

    # Testing: On the fly residual loss calculator
    def loss_func_otf(self, new_data):
        in0, in1, in2, in3, in4, in5, in6 = [torch.tensor(new_data[:, i:i+1]).float().to(device) for i in range(7)]  # Adjust indices based on new_data structure
        outf = torch.cat([in3, in4, in5, in6], dim=1)
        loss = physics(outf, in0, in1, in2, device)
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
    
    # Define input, hidden, and output layers
    input_features = config['layers']['input_features']
    hidden_layers = config['layers']['hidden_layers']
    hidden_width = config['layers']['hidden_width']
    output_features = config['layers']['output_features']
    layers = [input_features] + [hidden_width] * hidden_layers + [output_features]
    
    ########### Data for Fidelity ###########
    # Extract all data from csv file.
    data_fidelity_dir = config['data']['data_fidelity_dir']
    data_fidelity = np.genfromtxt(data_fidelity_dir, delimiter=' ').astype(np.float64) # load data

    # Create a dictionary to hold the different data columns
    data_fidelity_dict = {}
    data_fidelity_names = config['data']['data_fidelity_names']  # List of column names
    for i, name in enumerate(data_fidelity_names):
        data_fidelity_dict[name] = data_fidelity[:, i:i+1]
    
    # Create fidelity data by hstacking the values from the dictionary
    data_fidelity = np.hstack([data_fidelity_dict[key] for key in data_fidelity_names])

    # Normalization input data
    def normalize(data, data_min, data_max):
        return 2 * (data - data_min) / (data_max - data_min) - 1
    
    # Stacking and normalizing
    data_fidelity_names_normalize = config["data_normalize"]
    data_fidelity_training = []
    for key in data_fidelity_names:
        column = data_fidelity_dict[key]
        if key in data_fidelity_names_normalize:
            column_min = np.min(column)
            column_max = np.max(column)
            column = normalize(column, column_min, column_max)
        data_fidelity_training.append(column)
    data_fidelity_training = np.hstack(data_fidelity_training)

    # select training points randomly from the domain.
    Ntrain = config['Num_Training']
    idx = np.random.choice(data_fidelity_training.shape[0], Ntrain, replace=False)
    data_fidelity_training = data_fidelity_training[idx,:]

    ########### Data for Residual ###########
    # Create a dictionary to hold the different data columns
    data_residual_dict = {}
    data_residual_names = config['data']['data_residual_names']  # List of column names

    # also get some snapshots from FUNWAVE-TVD for residual loss
    funwave_dir = config['data']['funwave_dir']
    grid_nx = config['data']['grid_nx']
    grid_ny = config['data']['grid_ny']
    grid_dx = config['data']['grid_dx']
    grid_dy = config['data']['grid_dy']

    # select all available points (gauges) for residuals
    dx_interval = grid_dx*10
    dy_interval = grid_dy*10
    x_residual = np.arange(0, (grid_nx-1)*grid_dx + 1, grid_dx).astype(np.float64)
    y_residual = np.arange(0, (grid_ny-1)*grid_dy + 1, grid_dy).astype(np.float64)

    # meshgrid x and y and only keep selected points
    X_residual, Y_residual = np.meshgrid(x_residual, y_residual)
    X_residual = X_residual[::dx_interval, ::dy_interval]
    Y_residual = Y_residual[::dx_interval, ::dy_interval]

    # Flatten the X and Y arrays
    X_residual_flat = X_residual.flatten().reshape(-1, 1)
    Y_residual_flat = Y_residual.flatten().reshape(-1, 1)

    data_residual_training = np.empty((0, input_features))
    
    data_residual_snapshots = config["data_residual_snapshots"]

    for t_residual in data_residual_snapshots:
        
        file_suffix = str(t_residual).zfill(5)  # Pad the number with zeros to make it 5 digits
        
        T_residual = np.full((grid_nx, grid_ny), t_residual, dtype=np.float64)
        T_residual = T_residual[::dx_interval, ::dy_interval]
        T_residual_flat = T_residual.flatten().reshape(-1, 1)
        
        Z_residual = np.loadtxt(funwave_dir + f'/eta_{file_suffix}')
        Z_residual = Z_residual[::dx_interval, ::dy_interval]
        Z_residual_flat = Z_residual.flatten().reshape(-1, 1)

        # Create new data for this iteration
        new_data = np.hstack(
            (T_residual_flat, X_residual_flat, Y_residual_flat, Z_residual_flat)
             )
        # Append new data to X_f_star
        data_residual_training = np.vstack((data_residual_training, new_data))

    # Stacking and normalizing
    data_residual_names = config["data_residual_names"]
    data_residual_names_normalize = config["data_normalize"]
    data_residual_training = []
    for key in data_residual_names:
        column = data_residual_dict[key]
        if key in data_residual_names_normalize:
            column_min = np.min(column)
            column_max = np.max(column)
            column = normalize(column, column_min, column_max)
        data_residual_training.append(column)
    data_residual_training = np.hstack(data_residual_training)

    # set up the pinn model
    AdamIt = config['Adam_MaxIt']
    LBFGSIt = config['LBFGS_MaxIt']
    model = PINN(data_fidelity_training, data_residual_training, layers, AdamIt, LBFGSIt)
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))

