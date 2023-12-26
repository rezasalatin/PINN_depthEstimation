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
    def __init__(self, fidelity_input_train, fidelity_true_train, residual_input_train):
            
        # Define input, hidden, output layers, and DNN
        self.input_features = config['layers']['input_features']
        self.hidden_layers = config['layers']['hidden_layers']
        self.hidden_width = config['layers']['hidden_width']
        self.output_features = config['layers']['output_features']
        self.layers = [self.input_features] + [self.hidden_width] * self.hidden_layers + [self.output_features]
        self.dnn = DNN(self.layers).to(device)

        # Initialize the optimizers
        self.init_optimizers()
        # max iteration for optimizers
        self.adam_maxit = config['adam_optimizer']['max_it']

        # Initialize iteration counter
        self.iter = 0
        
        # Create individual attributes for each column in fidelity_input_train
        inputs = config['data_fidelity']['inputs']          
        for i, var_name in enumerate(inputs):
            setattr(self, f'fidelity_input_{i}', torch.tensor(fidelity_input_train[:, i:i+1]).float().to(device))

        outputs = config['data_fidelity']['outputs']   
        for i, var_name in enumerate(outputs):
            setattr(self, f'fidelity_true_{i}', torch.tensor(fidelity_true_train[:, i:i+1]).float().to(device))
        
        # temporal and spatial information for physics part
        inputs = config['data_residual']['inputs']  # Get the variable configuration
        for i, (var_name, var_info) in enumerate(inputs.items()):
            requires_grad = "true" in var_info["requires_grad"]
            setattr(self, f'residual_input_{i}', torch.tensor(residual_input_train[:, i:i+1], requires_grad=requires_grad).float().to(device))

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
        fidelity_inputs = [getattr(self, f'fidelity_input_{i}') for i in range(self.input_features)]
        fidelity_predictions = self.dnn(torch.cat(fidelity_inputs, dim=-1))
        
        # Dynamic fidelity outputs and loss calculation
        fidelity_loss = 0
        for i in range(self.output_features):
            pred = fidelity_predictions[:, i:i+1].to(device)
            true = getattr(self, f'fidelity_true_{i}')
            fidelity_loss += torch.mean((true - pred)**2)
        
       # Dynamic residual inputs
        residual_inputs = [getattr(self, f'residual_input_{i}') for i in range(self.input_features)]
        residual_outputs = self.dnn(torch.cat(residual_inputs, dim=-1))
        residual_loss = physics(residual_outputs, residual_inputs, device)
        # Total loss
        weight_fidelity = config['loss']['weight_fid_loss']
        weight_residual = config['loss']['weight_res_loss']
        loss = weight_fidelity * fidelity_loss + weight_residual * residual_loss
                
        # iteration (epoch) counter
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Epoch %d, fidelity_loss: %.5e, residual_loss: %.5e, Total Loss: %.5e' % 
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
    
    dir = config['data_fidelity']['dir']            # Data directory
    inputs = config['data_fidelity']['inputs']      # List of input variable names
    outputs = config['data_fidelity']['outputs']    # List of exact/output variable names

    # Create dictionaries for input and output data columns
    fidelity_input, fidelity_true = {}, {}
    # Extract all data from csv file.
    data = np.genfromtxt(dir, delimiter=' ', dtype=None, names=True, encoding=None)

    # Iterate over the columns and assign them to the respective dictionaries
    for key in data.dtype.names:
        if key in inputs:
            fidelity_input[key] = data[key]
        if key in outputs:
            fidelity_true[key] = data[key]

    # Normalize input data
    fidelity_input_min_max = op.get_min_max(fidelity_input)
    for key in fidelity_input:
        fidelity_input[key] = op.normalize(fidelity_input[key], fidelity_input_min_max[key][0], fidelity_input_min_max[key][1])

    # Single NumPy array from dictionaries
    fidelity_input = np.column_stack([fidelity_input[key] for key in inputs])
    fidelity_true = np.column_stack([fidelity_true[key] for key in outputs])

    # select n training points randomly from the domain.
    n_training = config['data_fidelity']['training_points']
    idx = np.random.choice(fidelity_input.shape[0], n_training, replace=False)
    fidelity_input_train = fidelity_input[idx,:]
    fidelity_true_train = fidelity_true[idx,:]

    #########################################
    ########### Data for Residual ###########
    #########################################

    dir = config['data_residual']['dir']
    inputs = config['data_residual']['inputs']
    outputs = config['data_residual']['outputs']
    
    residual_snaps = config['data_residual']['numerical_model_snapshots']
    interval_x = config['numerical_model']['interval_x']
    interval_y = config['numerical_model']['interval_y']

    residual_input_train = np.empty((0, len(inputs)))
    for i in residual_snaps:
        file_suffix = str(i).zfill(5)
        
        # Dictionary to store the loaded data
        residual_input = {}

        # Iterate over the mapping and load each file
        for key, value in inputs.items():
            
            file_name = value["file"]
            fname = file_name if key in ['x', 'y', 'h'] else f"{file_name}_{file_suffix}"    
            file_path = dir + fname
            residual_input[key] = np.loadtxt(file_path)
    
        residual_input = np.column_stack([residual_input[key].flatten() for key in inputs])
       
        # Concatenate the new array to the existing residual_in_train
        residual_input_train = np.vstack((residual_input_train, residual_input))

    # set up the pinn model
    model = pinn(fidelity_input_train, fidelity_true_train, residual_input_train)
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))

