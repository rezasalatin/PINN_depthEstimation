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
from physics import Boussinesq_simple as physics
import operations as op
from dnn import DNN

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

        # max iteration for optimizers
        self.AdamIt = AdamIt
        self.LBFGSIt = LBFGSIt

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
            lr = config['optimizer']['lbfgs_learning_rate'],
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
    
# Main
if __name__ == "__main__": 
    
    # Define input, hidden, and output layers
    input_features = config['layers']['input_features']
    hidden_layers = config['layers']['hidden_layers']
    hidden_width = config['layers']['hidden_width']
    output_features = config['layers']['output_features']
    layers = [input_features] + [hidden_width] * hidden_layers + [output_features]
    # Optimizer parameters
    adam_maxit = config['adam_optimizer']['max_it']
    lbfgs_maxit = config['lbfgs_optimizer']['max_it']

    #########################################
    ########### Data for Fidelity ###########
    # Extract all data from csv file.
    dir = config['data_fidelity']['dir']
    data = np.genfromtxt(dir, delimiter=' ', dtype=None, names=True, encoding=None)

    # Create dictionaries for input and output data columns
    fidelity_in, fidelity_out = {}, {}
    vars_in = config['data_fidelity']['data_in']    # List of input variable names
    vars_out = config['data_fidelity']['data_out']   # List of exact/output variable names

    # Iterate over the columns and assign them to the respective dictionaries
    for key in data.dtype.names:
        if key in vars_in:
            fidelity_in[key] = data[key]
        if key in vars_out:
            fidelity_out[key] = data[key]

    # Normalize input data
    in_min_max = op.get_min_max(fidelity_in)
    for key in fidelity_in:
        fidelity_in[key] = op.normalize(fidelity_in[key], in_min_max[key][0], in_min_max[key][1])

    # Single NumPy array from dictionaries
    fidelity_in_array = np.column_stack([fidelity_in[key] for key in vars_in])
    fidelity_out_array = np.column_stack([fidelity_out[key] for key in vars_out])

    # select n training points randomly from the domain.
    n_training = config['data_fidelity']['training_points']
    idx = np.random.choice(fidelity_in_array.shape[0], n_training, replace=False)
    fidelity_in_train = fidelity_in_array[idx,:]
    fidelity_out_train = fidelity_out_array[idx,:]

    #########################################
    ########### Data for Residual ###########
    # FUNWAVE-TVD snapshots for residual loss
    dir = config['data_residual']['dir']
    vars_in = config['data_resisdual']['data_in']
    vars_out = config['data_resisdual']['data_out']
    
    residual_snaps = config['data_residual']['numerical_model_snapshots']
    interval_x = config['data_residual']['interval_x']
    interval_y = config['data_residual']['interval_y']

    residual_in_train = []  # List to store the flattened data dictionaries

    for i in residual_snaps:
        file_suffix = str(i).zfill(5)
        
        # Dictionary to store the loaded data
        residual_in = {}

        # Iterate over the mapping and load each file
        for var_name, file_name in vars_in.items():

            # 'x', 'y', and 'h' are not changing with time
            if var_name in ['x', 'y', 'h']:
                key = file_name
            else:
                key = f"{file_name}_{file_suffix}"

            file_path = dir + key
            residual_in[key] = np.loadtxt(file_path)

            residual_in_array = np.column_stack([residual_in[key] for key in vars_in])
       
        # Append the flattened data dictionary for this snapshot to the list
        residual_in_train.append(residual_in_array)

    # set up the pinn model
    model = PINN(fidelity_in_train, fidelity_out_train, residual_in_train, layers, adam_maxit, lbfgs_maxit)
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))

