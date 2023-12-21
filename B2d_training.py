"""
PINN training for Boussinesq 1d
@author: Reza Salatin
December 2023
w/ Pytorch
"""

import torch
from collections import OrderedDict
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
import os

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
folder_name = now.strftime("log_%Y%m%d_%H%M%S")
# Create a directory with the folder_name
log_dir = f"../pinn_log/{folder_name}"
os.makedirs(log_dir, exist_ok=True)
    
# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            linear_layer = torch.nn.Linear(layers[i], layers[i+1])

            # Xavier initialization
            torch.nn.init.xavier_uniform_(linear_layer.weight)
            # Set the biases to zero
            torch.nn.init.zeros_(linear_layer.bias)

            layer_list.append(('layer_%d' % i, linear_layer))
            layer_list.append(('activation_%d' % i, self.activation()))

        # Apply Xavier initialization to the last layer as well
        last_linear_layer = torch.nn.Linear(layers[-2], layers[-1])
        torch.nn.init.xavier_uniform_(last_linear_layer.weight)

        # Initialize the biases of the last layer to zero
        torch.nn.init.zeros_(last_linear_layer.bias)

        layer_list.append(('layer_%d' % (self.depth - 1), last_linear_layer))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out
    
# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X_u, X_f, layers, X_min, X_max, AdamIt, LBFGSIt):
        
        # data
        self.t_u = torch.tensor(X_u[:, 0:1]).float().to(device)
        self.x_u = torch.tensor(X_u[:, 1:2]).float().to(device)
        self.y_u = torch.tensor(X_u[:, 2:3]).float().to(device)

        self.h_u = torch.tensor(X_u[:, 3:4]).float().to(device)
        self.z_u = torch.tensor(X_u[:, 4:5]).float().to(device)
        self.u_u = torch.tensor(X_u[:, 5:6]).float().to(device)
        self.v_u = torch.tensor(X_u[:, 6:7]).float().to(device)
        
        self.t_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_f[:, 2:3], requires_grad=True).float().to(device)                
        self.z_f = torch.tensor(X_f[:, 3:4]).float().to(device)
        
        self.layers = layers

        self.vals_min = torch.tensor(X_min).float().to(device)
        self.vals_max = torch.tensor(X_max).float().to(device)
        
        # deep neural networks
        self.dnn = DNN(layers).to(device)
        # initialize iteration number
        self.iter = 0
        # Adam and LBFGS Iterations
        self.AdamIt = AdamIt
        self.LBFGSIt = LBFGSIt

    # Initialize two optimizers
    def init_optimizers(self):
        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr=1e-5 # learning rate
        )

        # L-BFGS optimizer
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=self.LBFGSIt, 
            max_eval=self.LBFGSIt*1.25, 
            history_size=100,
            tolerance_grad=1e-5, 
            tolerance_change=1e-7,
            line_search_fn="strong_wolfe"
        )

        # Define learning rate scheduler for Adam
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size=10000, 
            gamma=0.8
        )
        
    def net_u(self, t, x, y, z):  

        # input normalization [-1, 1]
        t = 2.0 * (t - self.vals_min[0])/(self.vals_max[0]-self.vals_min[0]) - 1.0
        x = 2.0 * (x - self.vals_min[1])/(self.vals_max[1]-self.vals_min[1]) - 1.0
        y = 2.0 * (y - self.vals_min[2])/(self.vals_max[2]-self.vals_min[2]) - 1.0
        z = 2.0 * (z - self.vals_min[4])/(self.vals_max[4]-self.vals_min[4]) - 1.0
        
        output = self.dnn(torch.cat([t, x, y, z], dim=1))
        return output
    

    def physics_simple(self, output_f_pred):

        h_pred = output_f_pred[:, 0:1].to(device)
        z_pred = output_f_pred[:, 1:2].to(device)
        u_pred = output_f_pred[:, 2:3].to(device)
        v_pred = output_f_pred[:, 3:4].to(device)

        # This u is not correct. It is at a specific depth. For equations, calculate the u at the surface.

        u_t = self.compute_gradient(u_pred, self.t_f)
        u_x = self.compute_gradient(u_pred, self.x_f)
        u_y = self.compute_gradient(u_pred, self.y_f)

        v_t = self.compute_gradient(v_pred, self.t_f)
        v_x = self.compute_gradient(v_pred, self.x_f)
        v_y = self.compute_gradient(v_pred, self.y_f)

        z_t = self.compute_gradient(z_pred, self.t_f)
        z_x = self.compute_gradient(z_pred, self.x_f)
        z_y = self.compute_gradient(z_pred, self.y_f)

        # Higher orders (refer to Shi et al. 2012, Ocean Modeling)
        hu = h_pred * u_pred
        hv = h_pred * v_pred
        hu_x = self.compute_gradient(hu, self.x_f)
        hv_y = self.compute_gradient(hv, self.y_f)

        # loss with physics (Navier Stokes / Boussinesq etc)
        f_cont = z_t + hu_x + hv_y # continuity eq.
        f_momx = u_t + u_pred * u_x + v_pred * u_y + 9.81 * z_x   # momentum in X dir
        f_momy = v_t + u_pred * v_x + v_pred * v_y + 9.81 * z_y   # momentum in Y dir
        
        loss_f = torch.mean(f_cont**2) + torch.mean(f_momx**2) + torch.mean(f_momy**2)

        return loss_f
    
    def physics(self, output_f_pred):

        h_pred = output_f_pred[:, 0:1].to(device)
        z_pred = output_f_pred[:, 1:2].to(device)
        u_pred = output_f_pred[:, 2:3].to(device)
        v_pred = output_f_pred[:, 3:4].to(device)

        # This u is not correct. It is at a specific depth. For equations, calculate the u at the surface.

        u_t = self.compute_gradient(u_pred, self.t_f)
        u_x = self.compute_gradient(u_pred, self.x_f)
        u_y = self.compute_gradient(u_pred, self.y_f)

        v_t = self.compute_gradient(v_pred, self.t_f)
        v_x = self.compute_gradient(v_pred, self.x_f)
        v_y = self.compute_gradient(v_pred, self.y_f)

        z_t = self.compute_gradient(z_pred, self.t_f)
        z_x = self.compute_gradient(z_pred, self.x_f)
        z_y = self.compute_gradient(z_pred, self.y_f)

        # Higher orders (refer to Shi et al. 2012, Ocean Modeling)
        hu = h_pred * u_pred
        hv = h_pred * v_pred
        hu_x = self.compute_gradient(hu, self.x_f)
        hv_y = self.compute_gradient(hv, self.y_f)
        A = hu_x + hv_y
        B = u_x + v_y
        A_t = self.compute_gradient(A, self.t_f)
        A_x = self.compute_gradient(A, self.x_f)
        A_y = self.compute_gradient(A, self.y_f)
        B_t = self.compute_gradient(B, self.t_f)
        B_x = self.compute_gradient(B, self.x_f)
        B_y = self.compute_gradient(B, self.y_f)

        z_alpha = -0.53 * h_pred + 0.47 * z_pred
        z_alpha_x = self.compute_gradient(z_alpha, self.x_f)
        z_alpha_y = self.compute_gradient(z_alpha, self.y_f)

        # calculate u and v at the water surface elevation
        temp1 = (z_alpha**2/2-1/6*(h_pred**2-h_pred*z_pred+z_pred**2))
        temp2 = (z_alpha+1/2*(h_pred-z_pred))
        u_2 = temp1*B_x + temp2*A_x
        v_2 = temp1*B_y + temp2*A_y
        u_surface = u_pred + u_2
        v_surface = v_pred + v_2
        H = h_pred + z_pred
        Hu_surface = H*u_surface
        Hv_surface = H*v_surface
        Hu_x = self.compute_gradient(Hu_surface, self.x_f)
        Hv_y = self.compute_gradient(Hv_surface, self.y_f)

        # V1, dispersive Boussinesq terms
        V1Ax = z_alpha**2/2*B_x + z_alpha*A_x
        V1Ax_t = self.compute_gradient(V1Ax, self.t_f)
        V1Ay = z_alpha**2/2*B_y + z_alpha*A_y
        V1Ay_t = self.compute_gradient(V1Ay, self.t_f)
        V1B = z_pred**2/2*B_t+z_pred*A_t
        V1Bx = self.compute_gradient(V1B, self.x_f)
        V1By = self.compute_gradient(V1B, self.y_f)
        V1x = V1Ax_t - V1Bx
        V1y = V1Ay_t - V1By
        # V2, dispersive Boussinesq terms
        V2 = (z_alpha-z_pred)*(u_pred*A_x+v_pred*A_y) \
            +1/2*(z_alpha**2-z_pred**2)*(u_pred*B_x+v_pred*B_y)+1/2*(A+z_pred*B)**2
        V2x = self.compute_gradient(V2, self.x_f)
        V2y = self.compute_gradient(V2, self.y_f)
        # V3
        omega0 = v_x - u_y
        omega2 = z_alpha_x * (A_y + z_alpha * B_y) - z_alpha_y * (A_x + z_alpha * B_x)
        V3x = -omega0*v_2 - omega2*v_pred
        V3y = omega0*u_2 + omega2*u_pred

        # loss with physics (Navier Stokes / Boussinesq etc)
        f_cont = z_t + Hu_x + Hv_y # continuity eq.
        f_momx = u_t + u_pred * u_x + v_pred * u_y + 9.81 * z_x + V1x + V2x + V3x   # momentum in X dir
        f_momy = v_t + u_pred * v_x + v_pred * v_y + 9.81 * z_y + V1y + V2y + V3y   # momentum in Y dir
        
        loss_f = torch.mean(f_cont**2) + torch.mean(f_momx**2) + torch.mean(f_momy**2)

        return loss_f

    
    # compute gradients for the pinn
    def compute_gradient(self, pred, var):
        
        grad = torch.autograd.grad(
            pred, var, 
            grad_outputs=torch.ones_like(pred),
            retain_graph=True,
            create_graph=True
        )[0]
        return grad
    
    def loss_func(self):
        
        output_u_pred = self.net_u(self.t_u, self.x_u, self.y_u, self.z_u)
        
        h_pred = output_u_pred[:, 0:1].to(device)
        z_pred = output_u_pred[:, 1:2].to(device)
        u_pred = output_u_pred[:, 2:3].to(device)
        v_pred = output_u_pred[:, 3:4].to(device)
        
        weight_h = 1.0
        weight_z = 1.0
        weight_u = 1.0
        weight_v = 1.0

        loss_comp_h = torch.mean((self.h_u - h_pred)**2)
        loss_comp_z = torch.mean((self.z_u - z_pred)**2)
        loss_comp_u = torch.mean((self.u_u - u_pred)**2)
        loss_comp_v = torch.mean((self.v_u - v_pred)**2)

        loss_u = weight_h * loss_comp_h + weight_z * loss_comp_z \
            + weight_u * loss_comp_u + weight_v * loss_comp_v
        
        # Physics 
        output_f_pred = self.net_u(self.t_f, self.x_f, self.y_f, self.z_f)
        loss_f = self.physics_simple(output_f_pred)
        
        weight_loss_u = 1.0
        weight_loss_f = 1.0
        
        loss = weight_loss_u * loss_u + weight_loss_f * loss_f
                
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % 
                (self.iter, loss_u.item(), loss_f.item(), loss.item()))

        return loss
    
    def train(self):
        self.dnn.train()

        # First phase of training with Adam
        for i in range(self.AdamIt):  # number of iterations
            self.optimizer_Adam.zero_grad()  # Zero gradients for Adam optimizer
            loss = self.loss_func()
            loss.backward()
            self.optimizer_Adam.step()
            self.scheduler_Adam.step()  # Update the learning rate

        # Second phase of training with LBFGS
        def closure():
            self.optimizer_LBFGS.zero_grad()  # Zero gradients for LBFGS optimizer
            loss = self.loss_func()
            loss.backward()
            return loss
        
        self.optimizer_LBFGS.step(closure)

        # Print final loss after training
        final_loss = self.loss_func()  # Get the final loss
        print('Final Iter %d, Total Loss: %.5e' % (self.iter, final_loss.item()))
    
    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        z = torch.tensor(X[:, 3:4]).float().to(device)

        output_pred = self.net_u(t, x, y, z)
        
        h_pred = output_pred[:, 0:1].to(device)
        z_pred = output_pred[:, 1:2].to(device)
        u_pred = output_pred[:, 2:3].to(device)
        v_pred = output_pred[:, 3:4].to(device)

        return h_pred, z_pred, u_pred, v_pred  # Return the computed predictions
    
if __name__ == "__main__": 
       
    # Define training points + iterations for Adam and LBFGS
    Ntrain = 9600
    AdamIt = 1000
    LBFGSIt = 50000
    
    # Define input, hidden, and output layers
    input_features = 4 # t, x, y, eta
    hidden_layers = 20
    hidden_width = 20
    output_features = 4 # h, eta, u, v
    layers = [input_features] + [hidden_width] * hidden_layers + [output_features]
    
    funwave_dir = '../data/output_irr_pinn'
    x_grid = 251
    y_grid = 501
    dx = 2
    dy = 2
    
    training_data_dir = '../data/beach2d_irr.csv'
    
    # Extract all data from csv file.
    data = np.genfromtxt(training_data_dir, delimiter=' ').astype(np.float32) # load data
    t_all = data[:, 0:1].astype(np.float64)
    x_all = data[:, 1:2].astype(np.float64)
    y_all = data[:, 2:3].astype(np.float64)
    h_all = data[:, 3:4].astype(np.float64)
    z_all = data[:, 4:5].astype(np.float64)
    u_all = data[:, 5:6].astype(np.float64)
    v_all = data[:, 6:7].astype(np.float64)
    
    X_star = np.hstack((t_all, x_all, y_all, h_all, z_all, u_all, v_all))
    # get data range for data normalization for training only
    X_star_min = np.min(X_star, axis=0)
    X_star_max = np.max(X_star, axis=0)
    
    # select training points randomly from the domain.
    idx = np.random.choice(X_star.shape[0], Ntrain, replace=False)
    X_u_star = X_star[idx,:]
    
    # select all available points (gauges) for residuals
    dx_interval = dx*20
    dy_interval = dy*20
    X_f_star = np.empty((0, 4))  # Assuming you have 4 columns: T, X, Y, Z
    x_f = np.arange(0, (x_grid-1)*dx + 1, dx).astype(np.float64)
    y_f = np.arange(0, (y_grid-1)*dy + 1, dy).astype(np.float64)
    X_f, Y_f = np.meshgrid(x_f, y_f)
    X_f = X_f[::dx_interval, ::dy_interval]
    Y_f = Y_f[::dx_interval, ::dy_interval]
    # Flatten the X and Y arrays
    X_f_flat = X_f.flatten().reshape(-1, 1)
    Y_f_flat = Y_f.flatten().reshape(-1, 1)
    
    for t in [250, 302, 367, 299]:
        
        file_suffix = str(t).zfill(5)  # Pad the number with zeros to make it 5 digits
        
        T_f = np.full((x_grid, y_grid), t, dtype=np.float64)
        T_f = T_f[::dx_interval, ::dy_interval]
        T_f_flat = T_f.flatten().reshape(-1, 1)
        
        Z_f = np.loadtxt(funwave_dir + f'/eta_{file_suffix}')
        Z_f = Z_f[::dx_interval, ::dy_interval]
        Z_f_flat = Z_f.flatten().reshape(-1, 1)

        # Create new data for this iteration
        new_data = np.hstack((T_f_flat, X_f_flat, Y_f_flat, Z_f_flat))
        # Append new data to X_f_star
        X_f_star = np.vstack((X_f_star, new_data))

    # set up the pinn model
    model = PhysicsInformedNN(X_u_star, X_f_star, layers, X_star_min, X_star_max, AdamIt, LBFGSIt)
    model.init_optimizers()  # Initialize optimizers
    
    
    ###### Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the trained model
    torch.save(model.dnn.state_dict(), os.path.join(log_dir, 'model.ckpt')) # Only the parameters
    torch.save(model.dnn, os.path.join(log_dir, 'complete_model.pth'))      # Entire model

    ###### Testing

    current_avg = 60 # seconds or 5Tp
    window_size = 60
    t_final = 4879

    # initiate for current calculations
    U_all_pred = np.zeros((x_grid*y_grid, window_size))
    V_all_pred = U_all_pred
    U_all_test = U_all_pred
    V_all_test = U_all_pred

    x_test = np.arange(0, (x_grid-1)*dx + 1, dx).astype(np.float64)
    y_test = np.arange(0, (y_grid-1)*dy + 1, dy).astype(np.float64)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    # Flatten the X and Y arrays
    X_test_flat = X_test.flatten().reshape(-1, 1)
    Y_test_flat = Y_test.flatten().reshape(-1, 1)

    # get bathymetry file and flatten it
    h_test = np.loadtxt(funwave_dir + '/dep.out')
    h_test_flat = h_test.flatten().reshape(-1, 1)

    for t in range(0,t_final+1):

        T_test = np.full((x_grid, y_grid), t, dtype=np.float64)
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
        
        
        # for all figures
        fsize = 14
        x_limits = [150, 500]
        y_limits = [0, 1000]

        if t % 56 == 0:

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
            plt.savefig(f'../plots/UV_pred_{file_suffix}.png', dpi=300)
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
            plt.savefig(f'../plots/h_pred_{file_suffix}.png', dpi=300)
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
            plt.savefig(f'../plots/Current_pred_{file_suffix}.png', dpi=300)
            plt.tight_layout()
            plt.show()
            plt.close()
            
            # Print the current value of t
            print(f'Figures for t = {t} are plotted!')
        
        # Concatenate the predictions for saving
        # predictions = np.hstack([h_pred, z_pred, u_pred, v_pred])

        # Save to a file
        # np.savetxt(f'../pinn_data/predictions_{file_suffix}.txt', predictions, delimiter=',', header='h_pred,z_pred,u_pred', comments='')
