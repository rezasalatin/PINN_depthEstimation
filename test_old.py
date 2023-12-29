import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import operations as op

class pinn:
    def __init__(self, model_path, device):
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()  # Set the model to evaluation mode
        self.device = device
        
        # data
        self.input_vars = config['data_residual']['inputs']
        self.output_vars = config['data_residual']['outputs']

    def prepare_input_data(self, input_data):
        # Normalize input data
        input_min_max = op.get_min_max(input_data)
        for key in input_data:
            input_data[key] = op.normalize(input_data[key], input_min_max[key][0], input_min_max[key][1])
            
        input_data = np.column_stack([input_data[key].flatten() for key in input_data])
        return input_data

    def predict(self, input_data):
        # Convert input data to tensor and send to device
        input_tensor = torch.tensor(self.prepare_input_data(input_data)).float().to(device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        return predictions

if __name__ == "__main__":
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration if needed
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Define the path to the saved model
    model_dir = config['test']['model_path']
    model_path = os.path.join(model_dir, 'model.pth')
    
    # data
    input_vars = config['data_residual']['inputs']
    output_vars = config['data_residual']['outputs']
    # Setup for testing
    folder = config['numerical_model']['dir']
    grid_x = config['numerical_model']['nx']
    grid_y = config['numerical_model']['ny']
    dx = config['numerical_model']['dx']
    dy = config['numerical_model']['dy']
    file_no = config['numerical_model']['number_of_files']
    
    # Initialize the predictor
    predictor = pinn(model_path, device)
    
    for i in range(file_no):
        file_suffix = str(i).zfill(5)
        
        # Dictionary to store the loaded data
        inputs = {}

        # Iterate over the mapping and load each file
        for key, value in input_vars.items():
            
            file_name = value["file"]
            fname = file_name if key in ['x', 'y', 'h'] else f"{file_name}_{file_suffix}"    
            file_path = os.path.join(folder, fname)
            data = np.loadtxt(file_path)
            inputs[key] = data
        
        # Make predictions
        predictions = predictor.predict(inputs)
        
        for i, var_name in enumerate(output_vars):
            setattr(self, var_name, predictions[:, i:i+1])
    
'''
        
        h_pred = h_pred.detach().cpu().numpy()
        z_pred = z_pred.detach().cpu().numpy()
        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()        
    
        # Reshape predictions to match original grid shape
        h_pred_reshaped = h_pred.reshape(X_test.shape)
        z_pred_reshaped = z_pred.reshape(X_test.shape)
        u_pred_reshaped = u_pred.reshape(X_test.shape)
        v_pred_reshaped = v_pred.reshape(X_test.shape)
    
        # Moving average for current calculations
        U_all_pred[:, 0] = u_pred.flatten()
        V_all_pred[:, 0] = v_pred.flatten()
        U_all_test[:, 0] = U_test_flat.flatten()
        V_all_test[:, 0] = V_test_flat.flatten()
        U_avg_pred = np.mean(U_all_pred, axis=1)
        V_avg_pred = np.mean(V_all_pred, axis=1)
        U_avg_test = np.mean(U_all_test, axis=1)
        V_avg_test = np.mean(V_all_test, axis=1)
        U_avg_pred = U_avg_pred.reshape(X_test.shape)
        V_avg_pred = V_avg_pred.reshape(X_test.shape)
        U_avg_test = U_avg_test.reshape(X_test.shape)
        V_avg_test = V_avg_test.reshape(X_test.shape)
        # Roll the matrices to the right by one column
        U_all_pred = np.roll(U_all_pred, shift=1, axis=1)
        V_all_pred = np.roll(V_all_pred, shift=1, axis=1)
        U_all_test = np.roll(U_all_test, shift=1, axis=1)
        V_all_test = np.roll(V_all_test, shift=1, axis=1)
    
        ########## Model update during testing
        # Format the predictions to match the training data structure
        new_data = np.hstack((T_test_flat, X_test_flat, Y_test_flat, h_pred, z_pred, u_pred, v_pred))
        # Update the model using new predictions
        model.update_model_otf(new_data)
        
        # for all figures
        fsize = config['plot']['figure_size']
        x_limits = config['plot']['x_limits']
        y_limits = config['plot']['y_limits']
    
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
            # Plotting figure 1 for UV
            fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
            n = 10  # Interval: Sample every nth point (e.g., n=10 for every 10th grid point)
            scale = 25  # Arrow size: Adjust as needed for visibility
            # Sampling the grid and vector field
            X_sampled = X_test[::n, ::n]
            Y_sampled = Y_test[::n, ::n]
            U_pred_sampled = u_pred_reshaped[::n, ::n]
            V_pred_sampled = v_pred_reshaped[::n, ::n]
            U_test_sampled = U_test[::n, ::n]
            V_test_sampled = V_test[::n, ::n]
            # X, Y, UV plot with quivers and controlled intervals and arrow size
            ax.quiver(X_sampled, Y_sampled, U_test_sampled, V_test_sampled, color='black', scale=scale)
            ax.quiver(X_sampled, Y_sampled, U_pred_sampled, V_pred_sampled, color='red', alpha=0.5, scale=scale)
            ax.set_xlabel('X (m)', fontsize=fsize)
            ax.set_ylabel('Y (m)', fontsize=fsize)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            # Save the plot with file number in the filename
            plt.savefig(f'../plots/plots/UV_pred_{file_suffix}.png', dpi=300)
            plt.tight_layout()
            plt.show()
            plt.close()
    
            ########## Fig 3
            # Plotting figure 3 for eta
            fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
            # X, Y, Z plot
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
    
            ########## Fig 4
            # Plotting figure 1 for UV average
            fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
            n = 10  # Interval: Sample every nth point (e.g., n=10 for every 10th grid point)
            scale = 25  # Arrow size: Adjust as needed for visibility
            # Sampling the grid and vector field
            X_sampled = X_test[::n, ::n]
            Y_sampled = Y_test[::n, ::n]
            U_pred_sampled = U_avg_pred[::n, ::n]
            V_pred_sampled = V_avg_pred[::n, ::n]
            U_test_sampled = U_avg_test[::n, ::n]
            V_test_sampled = V_avg_test[::n, ::n]
            # X, Y, UV plot with quivers and controlled intervals and arrow size
            ax.quiver(X_sampled, Y_sampled, U_test_sampled, V_test_sampled, color='black', scale=scale)
            ax.quiver(X_sampled, Y_sampled, U_pred_sampled, V_pred_sampled, color='red', alpha=0.5, scale=scale)
            ax.set_xlabel('X (m)', fontsize=fsize)
            ax.set_ylabel('Y (m)', fontsize=fsize)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            # Save the plot with file number in the filename
            plt.savefig(f'../plots/plots/Current_pred_{file_suffix}.png', dpi=300)
            plt.tight_layout()
            plt.show()
            plt.close()
            
            # Print the current value of t
            print(f'Figures for t = {t} are plotted!')
        
        # Concatenate the predictions for saving
        # predictions = np.hstack([h_pred, z_pred, u_pred, v_pred])
    
        # Save to a file
        # np.savetxt(f'../pinn_data/predictions_{file_suffix}.txt', predictions, delimiter=',', header='h_pred,z_pred,u_pred', comments='')

'''