"""
PINN training for Boussinesq 2d
@author: Reza Salatin
@email: reza.salatin@whoi.edu
December 2023
w/ Pytorch
"""

import torch
import numpy as np
import pandas as pd
import time
import datetime
import os
import json
from physics import Boussinesq_simple as physics_loss_calculator
import operations as op
from dnn import DNN as DNN

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
    def __init__(self, fidelity_input, fidelity_true, residual_input):
            
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
        self.fidelity_input_vars = config['data_fidelity']['inputs']          
        for i, var_name in enumerate(self.fidelity_input_vars):
            setattr(self, f'fidelity_input_{var_name}', torch.tensor(fidelity_input[:, i:i+1]).float().to(device))

        self.fidelity_output_vars = config['data_fidelity']['outputs']   
        for i, var_name in enumerate(self.fidelity_output_vars):
            setattr(self, f'fidelity_true_{var_name}', torch.tensor(fidelity_true[:, i:i+1]).float().to(device))
        
        # temporal and spatial information for physics part
        self.residual_input_vars = config['data_residual']['inputs']
        for i, (var_name, var_info) in enumerate(self.residual_input_vars.items()):
            requires_grad = "true" in var_info["requires_grad"]
            setattr(self, f'residual_input_{var_name}', torch.tensor(residual_input[:, i:i+1], requires_grad=requires_grad).float().to(device))

        self.residual_output_vars = config['data_residual']['outputs']

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
        inputs = [getattr(self, f'fidelity_input_{var_name}') for i, var_name in enumerate(self.fidelity_input_vars)]
        inputs = torch.cat(inputs, dim=-1)
        predictions = self.dnn(inputs)
        # fidelity_loss = fidelity_loss_calculator(predictions, true, device)
        # Dynamic fidelity outputs and loss calculation
        fidelity_loss = 0
        for i, var_name in enumerate(self.fidelity_output_vars):
            pred = predictions[:, i:i+1].to(device)
            true = getattr(self, f'fidelity_true_{var_name}')
            fidelity_loss += torch.mean((true - pred)**2)
        
        # Dynamic residual inputs
        inputs = [getattr(self, f'residual_input_{var_name}') for i, var_name in enumerate(self.residual_input_vars)]
        for var_name in self.residual_input_vars:
            setattr(self, var_name, getattr(self, f'residual_input_{var_name}'))
            
        predictions = self.dnn(torch.cat(inputs, dim=-1))
        for i, var_name in enumerate(self.residual_output_vars):
            setattr(self, var_name, predictions[:, i:i+1])
                    
        residual_loss = physics_loss_calculator(self.t, self.x, self.y, self.h, self.z, self.u, self.v)
            
        # Total loss
        weight_fidelity = config['loss']['weight_fid_loss']
        weight_residual = config['loss']['weight_res_loss']
        loss = weight_fidelity * fidelity_loss + weight_residual * residual_loss
                
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
        
        if self.iter % 100 == 0:
            # Print the log values
            print(f'Epoch {self.iter}, Fidelity Loss: {fidelity_loss.item():.5e}, Residual Loss: {residual_loss.item():.5e}, Total Loss: {loss.item():.5e}')

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
    
    folder = config['data_fidelity']['dir']             # Data directory
    inputs = config['data_fidelity']['inputs']          # List of input variable names
    outputs = config['data_fidelity']['outputs']        # List of exact/output variable names

    # Create dictionaries for input and output data columns
    fidelity_input, fidelity_true = {}, {}
    # Extract all data from csv file.
    data = pd.read_csv(folder, delim_whitespace=True)
    data = data.round(3)

    # Iterate over the columns and assign them to the respective dictionaries
    for key in data.columns:
        if key in inputs:
            fidelity_input[key] = data[key].to_numpy()  # Convert to NumPy array if needed
        if key in outputs:
            fidelity_true[key] = data[key].to_numpy()  # Convert to NumPy array if needed

    # min and max to normalize fidelity input data
    input_min_max = op.get_min_max(fidelity_input, config)
    #for key in fidelity_input:
    #    fidelity_input[key] = op.normalize(fidelity_input[key], input_min_max[key][0], input_min_max[key][1])

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

    inputs = config['data_residual']['inputs']
    outputs = config['data_residual']['outputs']

    folder = config['numerical_model']['dir']
    residual_snaps = config['data_residual']['numerical_model_snapshots']

    interval_x, interval_y = config['numerical_model']['interval_x'], config['numerical_model']['interval_y']
    dx, dy = config['numerical_model']['dx'], config['numerical_model']['dy']
    x_min, x_max = config['numerical_model']['x_min'], config['numerical_model']['x_max']
    y_min, y_max = config['numerical_model']['y_min'], config['numerical_model']['y_max']
    dt = config['numerical_model']['dt']

    residual_input_train = np.empty((0, len(inputs)))

    x = np.linspace(x_min, x_max, num=config['numerical_model']['nx']).astype(np.float64)
    y = np.linspace(y_min, y_max, num=config['numerical_model']['ny']).astype(np.float64)
    X_test, Y_test = np.meshgrid(x, y)

    for i in residual_snaps:

        file_suffix = str(i).zfill(5)
        
        # Dictionary to store the loaded data
        residual_input = {}

        # Iterate over the mapping and load each file
        for key, value in inputs.items():
            
            file_name = value["file"]
            if key == 't':
                data = np.full(X_test.shape, i*dt, dtype=np.float64)
            elif key == 'x':
                data = X_test
            elif key == 'y':
                data = Y_test
            else:
                fname = file_name if key == 'h' else f"{file_name}_{file_suffix}"    
                file_path = os.path.join(folder, fname)
                data = np.loadtxt(file_path)
                data = data.round(3)

            residual_input[key] = data[::interval_x, ::interval_y]
            
            # Normalize residual input data
            #residual_input[key] = op.normalize(residual_input[key], input_min_max[key][0], input_min_max[key][1])
            
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

