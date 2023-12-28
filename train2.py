"""
PINN training for Boussinesq 2d
@author: Reza Salatin
@email: reza.salatin@whoi.edu
December 2023
w/ Pytorch
"""

import torch
import numpy as np
import time
import datetime
import os
import json
from physics import Boussinesq_simple as physics_loss_calculator
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

# Get current date and time
now = datetime.datetime.now()
folder_name = now.strftime("log_%Y%m%d_%H%M")
# Create a directory with the folder_name
log_dir = os.path.join("..", "log", folder_name)
os.makedirs(log_dir, exist_ok=True)
    
# The physics-guided neural network
class PINN:
    # Initializer
    def __init__(self, fidelity_input, fidelity_true, residual_input):
        # Model setup
        self.setup_model()
        self.init_optimizers()

        # Prepare data
        self.prepare_fidelity_data(fidelity_input, fidelity_true)
        self.prepare_residual_data(residual_input)

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Device in use: GPU")
        else:
            device = torch.device('cpu')
            print("Device in use: CPU")
        return device

    def setup_model(self):
        # Define input, hidden, output layers, and DNN
        self.layers = [config['layers']['input_features']] + \
                      [config['layers']['hidden_width']] * config['layers']['hidden_layers'] + \
                      [config['layers']['output_features']]
        self.dnn = DNN(self.layers).to(device)

    def init_optimizers(self):
        # Adam optimizer setup
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr=config['adam_optimizer']['learning_rate']
        )
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size=config['adam_optimizer']['scheduler_step_size'], 
            gamma=config['adam_optimizer']['scheduler_gamma']
        )
        # L-BFGS optimizer setup
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=config['lbfgs_optimizer']['learning_rate'],
            max_iter=config['lbfgs_optimizer']['max_it'], 
            max_eval=config['lbfgs_optimizer']['max_evaluation'], 
            history_size=config['lbfgs_optimizer']['history_size'],
            tolerance_grad=config['lbfgs_optimizer']['tolerance_grad'],
            tolerance_change=config['lbfgs_optimizer']['tolerance_change'],
            line_search_fn=config['lbfgs_optimizer']['line_search_fn']
        )    

    def prepare_fidelity_data(self, fidelity_input, fidelity_true):
        self.fidelity_input = torch.tensor(fidelity_input).float().to(device)
        self.fidelity_true = torch.tensor(fidelity_true).float().to(device)

    def prepare_residual_data(self, residual_input):
        self.residual_input = torch.tensor(residual_input).float().to(device)
        self.residual_input.requires_grad_(True)

    def loss_func(self):
        # Fidelity loss
        fidelity_predictions = self.dnn(self.fidelity_input)
        fidelity_loss = torch.mean((self.fidelity_true - fidelity_predictions) ** 2)

        # Residual loss
        residual_predictions = self.dnn(self.residual_input)
        residual_loss = physics_loss_calculator(self.residual_input, residual_predictions)

        # Total loss
        weight_fidelity = config['loss']['weight_fid_loss']
        weight_residual = config['loss']['weight_res_loss']
        loss = weight_fidelity * fidelity_loss + weight_residual * residual_loss

        # Logging
        self.log_loss(fidelity_loss, residual_loss, loss)

        return loss

    def log_loss(self, fidelity_loss, residual_loss, total_loss):
        log_file_path = os.path.join(log_dir, 'log.txt')
        with open(log_file_path, 'a') as log_file:
            if os.stat(log_file_path).st_size == 0:
                log_file.write('Epoch, Fidelity Loss, Residual Loss, Total Loss\n')
            log_file.write(f'{self.iter}, {fidelity_loss.item():.5e}, {residual_loss.item():.5e}, {total_loss.item():.5e}\n')

    def train(self):
        self.dnn.train()
        # Adam optimizer training
        for _ in range(config['adam_optimizer']['max_it']):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_Adam.step()
            self.scheduler_Adam.step()

        # L-BFGS optimizer training
        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss_func()
            if loss.requires_grad:
                loss.backward()
            return loss
        self.optimizer_LBFGS.step(closure)

# Main execution
if __name__ == "__main__":

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        exit(1)

    # Data loading and preprocessing steps (as in your provided code)
    # ...

    # Initialize and train the model
    model = PINN(fidelity_input_train, fidelity_true_train, residual_input_train)
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f seconds' % elapsed)

    # Save the trained model
    torch.save(model.dnn, os.path.join(log_dir, 'model.pth'))
