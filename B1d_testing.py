"""
PINN testing for Boussinesq 1d
@author: Reza Salatin
December 2023
w/ Pytorch
"""

import torch
import numpy as np

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


# Define the model architecture parameters (as used during training)
input_features = 3
hidden_layers = 20
hidden_width = 20
output_features = 3
layers = [input_features] + [hidden_width] * hidden_layers + [output_features]

# Placeholder for minimum and maximum values of the dataset for normalization
# Replace these with the actual min and max values from your training dataset
X_star_min = [0, 0, 0]  # Example values, replace with actual
X_star_max = [1, 1, 1]  # Example values, replace with actual

# Initialize the model
model = PhysicsInformedNN(None, None, layers, X_star_min, X_star_max).to(device)

# Load the trained model weights
model.dnn.load_state_dict(torch.load('./log/model.ckpt', map_location=device))
model.dnn.eval()  # Set the model to evaluation mode

# Testing
X_test = np.arange(1024).reshape(-1, 1).astype(np.float64)  # Example test data

for t in range(200, 401):
    T_test = np.full((1024, 1), t).astype(np.float64)  # Time steps for testing

    file_suffix = str(t).zfill(5)  # Pad the number with zeros to make it 5 digits
    file_name = f'../funwave/eta_{file_suffix}'  # Construct the file name
    Z_data = np.loadtxt(file_name)
    Z_test = Z_data[1, :]  # Select the second row
    Z_test = Z_test.reshape(1024, 1)
    
    X_star = np.hstack((T_test, X_test, Z_test))  # Combine test data
    
    # Predict using the model
    h_pred, z_pred, u_pred = model.predict(X_star)

    # Process predictions as needed, e.g., save them to a file or analyze them
    # ...
