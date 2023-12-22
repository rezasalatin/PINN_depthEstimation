import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = os.path.join('../log/', 'model.pth')  # Adjust the path as needed
model = torch.load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

# Setup for testing
funwave_dir = config['data']['funwave_dir']
grid_x = config['data']['grid_x']
grid_y = config['data']['grid_y']
dx = config['data']['dx']
dy = config['data']['dy']

current_avg = 60 # seconds or 5Tp
window_size = 60
t_final = 500

# initiate for current calculations
    U_all_pred = np.zeros((grid_x*grid_y, window_size))
    V_all_pred = U_all_pred
    U_all_test = U_all_pred
    V_all_test = U_all_pred

    x_test = np.arange(0, (grid_x-1)*dx + 1, dx).astype(np.float64)
    y_test = np.arange(0, (grid_y-1)*dy + 1, dy).astype(np.float64)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    # Flatten the X and Y arrays
    X_test_flat = X_test.flatten().reshape(-1, 1)
    Y_test_flat = Y_test.flatten().reshape(-1, 1)

    # get bathymetry file and flatten it
    h_test = np.loadtxt(funwave_dir + '/dep.out')
    h_test_flat = h_test.flatten().reshape(-1, 1)

    for t in range(0,t_final+1):

        T_test = np.full((grid_x, grid_y), t, dtype=np.float64)
        T_test_flat = T_test.flatten().reshape(-1, 1)

        file_suffix = str(t).zfill(5)  # Pad the number with zeros to make it 5 digits
        Z_test = np.loadtxt(funwave_dir + f'/eta_{file_suffix}')  # Construct the file name
        Z_test_flat = Z_test.flatten().reshape(-1, 1)
        
        U_test = np.loadtxt(funwave_dir + f'/u_{file_suffix}')
        U_test_flat = U_test.flatten().reshape(-1, 1)

        V_test = np.loadtxt(funwave_dir + f'/v_{file_suffix}')
        V_test_flat = V_test.flatten().reshape(-1, 1)

        # make inputs ready for NN
        X_star = np.hstack((T_test_flat, X_test_flat, Y_test_flat, Z_test_flat))

        # feed into NN and get outpus
        h_pred, z_pred, u_pred, v_pred = model.predict(X_star)
        
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
