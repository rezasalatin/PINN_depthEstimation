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
from scipy.io import loadmat
import time
import datetime
import os
import json
from physics import physics_equation as physics_loss_calculator
import operations as op
from dnn import DNN
import plots

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
with open('config_CMB.json', 'r') as f:
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
    def __init__(self, fidelity_input, fidelity_true, residual_input):
            
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
        
        # Create individual attributes for each column in fidelity_input_train
        self.fidelity_input_vars = config['data_fidelity']['inputs']          
        for i, key in enumerate(self.fidelity_input_vars):
            setattr(self, f'fidelity_input_{key}', torch.tensor(fidelity_input[:, i:i+1]).float().to(device))

        self.fidelity_output_vars = config['data_fidelity']['outputs']   
        for i, key in enumerate(self.fidelity_output_vars):
            setattr(self, f'fidelity_true_{key}', torch.tensor(fidelity_true[:, i:i+1]).float().to(device))
        
        # temporal and spatial information for physics part
        self.residual_input_vars = config['data_residual']['inputs']
        for i, (key, info) in enumerate(self.residual_input_vars.items()):
            requires_grad = "true" in info["requires_grad"]
            setattr(self, f'residual_input_{key}', torch.tensor(residual_input[:, i:i+1], requires_grad=requires_grad).float().to(device))

        self.residual_output_vars = config['data_residual']['outputs']
        # Loss function weights
        self.weight_fidelity = config['loss']['weight_fid_loss']
        self.weight_residual = config['loss']['weight_res_loss']
        for key in self.residual_output_vars:
            setattr(self, f'weight_{key}', config['loss'][f'weight_{key}_loss'])



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
        inputs = [getattr(self, f'fidelity_input_{key}') for i, key in enumerate(self.fidelity_input_vars)]
        inputs = torch.cat(inputs, dim=-1)
        predictions = self.dnn(inputs)
        # fidelity_loss = fidelity_loss_calculator(predictions, true, device)
        # Dynamic fidelity outputs and loss calculation
        fidelity_loss = 0
        for i, key in enumerate(self.fidelity_output_vars):
            pred = predictions[:, i:i+1].to(device)
            true = getattr(self, f'fidelity_true_{key}')
            weight = getattr(self, f'weight_{key}')
            fidelity_loss += weight * torch.mean((true - pred)**2)
        
        # Dynamic residual inputs
        inputs = [getattr(self, f'residual_input_{key}') for i, key in enumerate(self.residual_input_vars)]
        for key in self.residual_input_vars:
            setattr(self, key, getattr(self, f'residual_input_{key}'))
                        
        predictions = self.dnn(torch.cat(inputs, dim=-1))
        for i, key in enumerate(self.residual_output_vars):
            setattr(self, key, predictions[:, i:i+1])
            
        #self.z_pred = self.z.clone().detach().cpu().numpy()
                    
        residual_loss = physics_loss_calculator(self.x, self.y, self.h, self.U, self.V, self.eta_mean, self.Hrms, self.k)
            
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
    
# Main
if __name__ == "__main__": 

    #########################################
    ########### Data for Fidelity ###########
    #########################################
    
    file = config['data_fidelity']['file']              # Data directory
    inputs = config['data_fidelity']['inputs']          # List of input variable names
    outputs = config['data_fidelity']['outputs']        # List of exact/output variable names

    # Create dictionaries for input and output data columns
    fidelity_input, fidelity_true = {}, {}

    # Extract all data from csv file.
    data = pd.read_csv(file)
    data = data.round(3)

    # Iterate over the columns and assign them to the respective dictionaries
    for key in data.columns:
        if key in inputs:
            fidelity_input[key] = data[key].to_numpy()  # Convert to NumPy array if needed
        if key in outputs:
            fidelity_true[key] = data[key].to_numpy()  # Convert to NumPy array if needed

    # min and max to normalize fidelity input data
    input_min_max = op.get_min_max(fidelity_input, config)
    for key in fidelity_input:
        fidelity_input[key] = op.normalize(fidelity_input[key], input_min_max[key][0], input_min_max[key][1])

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

    file = config['data_residual']['file']
    residual_snaps = config['data_residual']['snapshots']
    interval_x, interval_y = config['data_residual']['interval_x'], config['data_residual']['interval_y']
    
    # Dictionary to store the loaded data
    residual_input = {}
    residual_input_train = np.empty((0, 1))

    for key in inputs:
        data = loadmat(file, variable_names=key)
        #residual_input[key] = data[key][::interval_x, ::interval_y, residual_snaps]
        residual_input[key] = data[key][::interval_x, ::interval_y]
        del data
        residual_input[key] = op.normalize(residual_input[key], input_min_max[key][0], input_min_max[key][1])
        
        # Flatten and reshape to ensure it's a column vector
        residual_input_tmp = residual_input[key].reshape(-1, residual_input[key].shape[1])
        residual_input_tmp = np.transpose(residual_input_tmp)
        residual_input_tmp = residual_input_tmp.reshape(-1, 1)
    
        # Concatenate the new array to the existing residual_input_train
        if residual_input_train.size == 0:
            residual_input_train = residual_input_tmp
        else:
            residual_input_train = np.hstack((residual_input_train, residual_input_tmp))
            
    # Remove rows where one of the variables is NaN
    mask = np.isnan(residual_input_train).any(axis=1)
    residual_input_train = residual_input_train[~mask]

    # set up the pinn model
    model = pinn(fidelity_input_train, fidelity_true_train, residual_input_train)
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))

