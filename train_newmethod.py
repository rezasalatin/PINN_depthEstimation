"""
PINN training for Boussinesq 2d
@author: Reza Salatin
@email: reza.salatin@whoi.edu
December 2023
w/ Pytorch
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import time
import datetime
import os
import json
from physics import continuity_only as physics_loss_calculator
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
with open('config_CMB_h.json', 'r') as f:
    config = json.load(f)

# Get current date and time
now = datetime.datetime.now()
folder_name = now.strftime("%Y%m%d_%H%M")
# Create a directory with the folder_name
log_dir = f"../log/{folder_name}"
os.makedirs(log_dir, exist_ok=True)
    
# the physics-guided neural network
class pinn():

    # Initializer
    def __init__(self, data_input, data_true):
            
        # Define input, hidden, output layers, and DNN
        self.input_features = config['layers']['input_features']
        self.hidden_layers = config['layers']['hidden_layers']
        self.hidden_width = config['layers']['hidden_width']
        self.output_features = config['layers']['output_features']
        self.layers = [self.input_features] + [self.hidden_width] * self.hidden_layers + [self.output_features]

        # Define dropout rate
        self.dropout_rate = config['layers']['dropout_rate']

        # Initialization method
        self.init_type = config['layers']['init_type']

        # Initialize DNN
        self.dnn = DNN(self.layers, self.dropout_rate, self.init_type).to(device)

        # Initialize the optimizers
        self.init_optimizers()
        # max iteration for optimizers
        self.adam_maxit = config['adam_optimizer']['max_it']

        # Initialize iteration counter
        self.iter = 0
        
        # Create individual attributes
        self.input_vars = config['data']['inputs']          
        for i, (key, info) in enumerate(self.input_vars.items()):
            requires_grad = "true" in info["requires_grad"]
            setattr(self, f'input_{key}', torch.tensor(data_input[:, i:i+1], requires_grad=requires_grad).float().to(device))

        self.true_vars = config['data']['trues']   
        for i, key in enumerate(self.true_vars):
            setattr(self, f'true_{key}', torch.tensor(data_true[:, i:i+1]).float().to(device))
        
        self.unknown_vars = config['data']['unknowns']   

        # Loss function weights
        self.weight_fidelity = config['loss']['weight_fid_loss']
        self.weight_residual = config['loss']['weight_res_loss']

    # Initialize optimizers
    def init_optimizers(self):

        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr = config['adam_optimizer']['learning_rate']
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
        
        # inputs
        inputs = [getattr(self, f'input_{key}') for i, key in enumerate(self.input_vars)]
        inputs = torch.cat(inputs, dim=-1)
        # Make predictions
        predictions = self.dnn(inputs)

        # Fidelity loss
        fidelity_loss = 0
        for i, key in enumerate(self.true_vars):
            pred = predictions[:, i:i+1].to(device)
            true = getattr(self, f'true_{key}')
            fidelity_loss += F.mse_loss(pred, true)
        
        # Residual loss
        for i, key in enumerate(self.true_vars):
            setattr(self, f'pred_{key}', predictions[:, i:i+1])
        for j, key in enumerate(self.unknown_vars):
            setattr(self, f'pred_{key}', predictions[:, j+i+1:j+i+2])
            
        if self.iter == 50000:
            # Create a dictionary to store the pred_key values
            data_to_save = {}
            for key in self.true_vars + self.unknown_vars:
                tensor = getattr(self, f'pred_{key}')
                # Move the tensor from GPU to CPU
                tensor_cpu = tensor.cpu().detach().numpy()
                data_to_save[f'pred_{key}'] = tensor_cpu
            # Specify the filename where you want to save the MATLAB file
            matlab_filename = 'data_at50k.mat'
            # Save the data to the MATLAB file
            sio.savemat(matlab_filename, data_to_save)
            print(f'Data saved to {matlab_filename} after 50,000 iterations.')
            
            
        residual_loss = physics_loss_calculator(self.input_x, self.input_y, self.pred_h, self.pred_U, self.pred_V)
            
        # Total loss
        loss = self.weight_fidelity * fidelity_loss + self.weight_residual * residual_loss
                
        # iteration (epoch) counter
        self.iter += 1
        
        log_file_path = os.path.join(log_dir, 'log.txt')
        
        # Check if the file exists and is empty; if so, write the header
        if not os.path.exists(log_file_path) or os.stat(log_file_path).st_size == 0:
           with open(log_file_path, 'w') as log_file:
               log_file.write('Epoch, Fidelity Loss, Residual Loss, Total Loss\n')
               
        # Create a formatted string for the log values
        log_values = f'{self.iter}, {fidelity_loss.item():.5e}, {residual_loss.item():.5e}, {loss.item():.5e}\n'
        # Write the log values to a file
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_values)
        
        if self.iter % 1000 == 0:
            # Print the log values
            print(f'Epoch {self.iter}, Fidelity Loss: {fidelity_loss.item():.5e}, Residual Loss: {residual_loss.item():.5e}, Total Loss: {loss.item():.5e}')
            
        if self.iter <= 45000:
            if self.iter % 10000 == 0:
                # Save the trained model
                torch.save(self.dnn, os.path.join(log_dir, f'model_{self.iter}.pth'))
        else:
            if self.iter % 1000 == 0:
                # Save the trained model
                torch.save(self.dnn, os.path.join(log_dir, f'model_{self.iter}.pth'))
            
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
    
### Main
if __name__ == "__main__": 

    ## Input Data
    
    data_file = config['data']['file']             # Data directory
    input_vars = config['data']['inputs']           # List of input variable names
    true_vars = config['data']['trues']             # List of variable names we have have their true values
    unknown_vars = config['data']['unknowns']       # List of variable names we don't have their values

    # Create dictionaries for input and output data columns
    data_input, data_true = {}, {}
    data_input_train, data_true_train = np.empty((0, 1)), np.empty((0, 1))

    # Inputs
    for key in input_vars:
        data = loadmat(data_file, variable_names=key)
        data_input[key] = data[key]
        del data
        # min and max to normalize fidelity input data
        minmax = op.get_min_max(data_input[key], key, config)
        data_input[key] = op.normalize(data_input[key], minmax[key][0], minmax[key][1])
    
        # Concatenate the new array
        if data_input_train.size == 0:
            data_input_train = data_input[key]
        else:
            data_input_train = np.hstack((data_input_train, data_input[key]))

    # Trues
    for key in true_vars:
        data = loadmat(data_file, variable_names=key)
        data_true[key] = data[key]
        del data
    
        # Concatenate the new array
        if data_true_train.size == 0:
            data_true_train = data_true[key]
        else:
            data_true_train = np.hstack((data_true_train, data_true[key]))

    # Remove rows where one of the variables is NaN
    mask = np.isnan(data_true_train).any(axis=1)
    data_input_train = data_input_train[~mask]
    data_true_train = data_true_train[~mask]
    
    # Delete redundant data
    del data_file, data_input, data_true, input_vars, key, mask, minmax
    del now, true_vars, unknown_vars, folder_name

    # set up the pinn model
    model = pinn(data_input_train, data_true_train)

    ## Train
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))

