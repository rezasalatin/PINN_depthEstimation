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
class pinn():

    # Initializer
    def __init__(self, fidelity_in_train, fidelity_out_train, residual_in_train):
        
        # Create individual attributes for each column in fidelity_in_train
        num_columns = fidelity_in_train.shape[1]
        for i in range(num_columns):
            setattr(self, f'fidelity_in{i}', torch.tensor(fidelity_in_train[:, i:i+1]).float().to(device))
        
        # temporal and spatial information for physics part
        vars_in = config['data_residual']['data_in']  # Get the variable configuration
        num_columns = residual_in_train.shape[1]
        for i, (var_name, var_info) in enumerate(vars_in.items()):
            # Determine if the variable is differentiable
            requires_grad = "true" in var_info["requires_grad"]
            # Set the tensor with the appropriate requires_grad flag
            setattr(self, f'residual_in{i}', torch.tensor(residual_in_train[:, i:i+1], requires_grad=requires_grad).float().to(device))
            
        # Define input, hidden, output layers, and DNN
        self.input_features = config['layers']['input_features']
        self.hidden_layers = config['layers']['hidden_layers']
        self.hidden_width = config['layers']['hidden_width']
        self.output_features = config['layers']['output_features']
        self.layers = [self.input_features] + [self.hidden_width] * self.hidden_layers + [self.output_features]
        self.dnn = DNN(self.layers).to(device)

        # max iteration for optimizers
        self.adam_maxit = config['adam_optimizer']['max_it']

        # Initialize iteration counter
        self.iter = 0

    # Initialize optimizers
    def init_optimizers(self):

        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr = config['adam_optimizer']['learning_rate'], # learning rate
        )

        # Define learning rate scheduler for Adam
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size = config['adam_optimizer']['scheduler_step_size'], 
            gamma = config['adam_optimizer']['scheduler_gamma']
        )

        # L-BFGS optimizer
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr = config['lbfgs_optimizer']['learning_rate'],
            max_iter = config['lbfgs_optimizer']['max_it'], 
            max_eval = config['lbfgs_optimizer']['max_evaluation'], 
            history_size = config['lbfgs_optimizer']['history_size'],
            tolerance_grad = config['lbfgs_optimizer']['tolerance_grad'],
            tolerance_change = config['lbfgs_optimizer']['tolerance_change'],
            line_search_fn = config['lbfgs_optimizer']['line_search_fn']
        )    

    # Loss function
    def loss_func(self):
        
        # Dynamic fidelity inputs
        fidelity_inputs = [getattr(self, f'fidelity_in{i}') for i in range(self.num_fidelity_inputs)]
        fidelity_outputs = self.dnn(*fidelity_inputs)
        
        # Dynamic fidelity outputs and loss calculation
        fidelity_loss = 0
        for i in range(self.num_fidelity_outputs):
            fid_out_i = fidelity_outputs[:, i:i+1].to(device)
            fid_exact_i = getattr(self, f'fid_exact{i}')
            fid_loss_i = torch.mean((fid_exact_i - fid_out_i)**2)
            weight_fid_loss_i = config['loss'][f'weight_fid_loss{i}']
            fidelity_loss += weight_fid_loss_i * fid_loss_i
        
       # Dynamic residual inputs
        residual_inputs = [getattr(self, f'residual_in{i}') for i in range(self.num_residual_inputs)]
        residual_outputs = self.dnn(*residual_inputs)
        residual_loss = physics(residual_outputs, *residual_inputs, device)
    
        # Total loss
        weight_fidelity = config['loss']['weight_fidelity']
        weight_residual = config['loss']['weight_residual']
        loss = weight_fidelity * fidelity_loss + weight_residual * residual_loss
                
        # iteration (epoch) counter
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % 
                (self.iter, fidelity_loss.item(), residual_loss.item(), loss.item()))

        return loss

    
    # Model training with two optimizers
    def train(self):
        self.dnn.train()
        # Training with Adam optimizer
        for i in range(self.adam_maxit):
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

    #########################################
    ########### Data for Fidelity ###########
    #########################################
    
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
    #########################################

    # FUNWAVE-TVD snapshots for residual loss
    dir = config['data_residual']['dir']
    vars_in = config['data_residual']['data_in']
    vars_out = config['data_residual']['data_out']
    
    residual_snaps = config['data_residual']['numerical_model_snapshots']
    interval_x = config['numerical_model']['interval_x']
    interval_y = config['numerical_model']['interval_y']

    residual_in_train = np.empty((0, len(vars_in)))  # Initialize as empty array with appropriate columns

    for i in residual_snaps:
        file_suffix = str(i).zfill(5)
        
        # Dictionary to store the loaded data
        residual_in = {}

        # Iterate over the mapping and load each file
        for key, value in vars_in.items():
            
            file_name = value["file"]  # Extract the file name
    
            # Determine the filename, considering time variation
            fname = file_name if key in ['x', 'y', 'h'] else f"{file_name}_{file_suffix}"
    
            file_path = dir + fname
            residual_in[key] = np.loadtxt(file_path)
    
        residual_in_array = np.column_stack([residual_in[key].flatten() for key in vars_in])
       
        # Concatenate the new array to the existing residual_in_train
        residual_in_train = np.vstack((residual_in_train, residual_in_array))

    # set up the pinn model
    model = pinn(fidelity_in_train, fidelity_out_train, residual_in_train)
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))

