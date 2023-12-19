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
        self.z_f = torch.tensor(X_f[:, 4:5]).float().to(device)
        
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

        h_pred = output_f_pred[:, 0:1].to(device)
        z_pred = output_f_pred[:, 1:2].to(device)
        u_pred = output_f_pred[:, 2:3].to(device)
        v_pred = output_f_pred[:, 3:4].to(device)
            
        u_t = self.compute_gradient(u_pred, self.t_f)
        u_x = self.compute_gradient(u_pred, self.x_f)
        u_y = self.compute_gradient(u_pred, self.y_f)

        v_t = self.compute_gradient(v_pred, self.t_f)
        v_x = self.compute_gradient(v_pred, self.x_f)
        v_y = self.compute_gradient(v_pred, self.y_f)

        z_t = self.compute_gradient(z_pred, self.t_f)
        z_x = self.compute_gradient(z_pred, self.x_f)
        z_y = self.compute_gradient(z_pred, self.y_f)

        # loss with physics (Navier Stokes / Boussinesq etc)
        f_1 = u_t + (u_pred * u_x + v_pred * u_y) + 9.81 * (z_x + z_y)
        f_2 = v_t + (u_pred * v_x + v_pred * v_y) + 9.81 * (z_x + z_y)
        f_3 = z_t + (u_pred * z_x + v_pred * z_y) + (h_pred + z_pred) * (u_x + v_y)
        
        loss_f = torch.mean(f_1**2) + torch.mean(f_2**2) + torch.mean(f_3**2)
        
        weight_loss_u = 1.0
        weight_loss_f = 1.0
        
        loss = weight_loss_u * loss_u + weight_loss_f * loss_f
                
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % 
                (self.iter, loss_u.item(), loss_f.item(), loss.item()))
            print(
                'Loss_h: %.5e, Loss_z: %.5e, Loss_u: %.5e' % 
                (loss_comp_h.item(), loss_comp_z.item(), loss_comp_u.item()))

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
    Ntrain = 10000
    AdamIt = 10000
    LBFGSIt = 50000
    
    # Define input, hidden, and output layers
    input_features = 4 # t, x, y, eta
    hidden_layers = 5
    hidden_width = 8
    output_features = 4 # h, eta, u, v
    layers = [input_features] + [hidden_width] * hidden_layers + [output_features]
    
    # Extract all data from csv file.
    data = np.genfromtxt('../pinn_data/beach_2d_dt01.csv', delimiter=' ').astype(np.float32) # load data
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
    X_f_star = X_star
    
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

    funwave_dir = '../funwave2d_outputs'

    x_test = np.arange(0, 501, 2).astype(np.float64)
    y_test = np.arange(0, 1001, 2).astype(np.float64)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    # Flatten the X and Y arrays
    X_test_flat = X_test.flatten().reshape(-1, 1)
    Y_test_flat = Y_test.flatten().reshape(-1, 1)

    # get bathymetry file and flatten it
    h_test = np.loadtxt(funwave_dir + '/dep.out')
    h_test_flat = h_test.flatten().reshape(-1, 1)

    for t in range(200,251):

        T_test = np.full((501, 251), t, dtype=np.float64)
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
        
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        x_limits = [150, 500]
        y_limits = [0, 1000]

        # X, Y, Z plot
        cmap1 = axs[0].pcolor(X_test, Y_test, z_pred_reshaped, shading='auto')
        cbar1 = fig.colorbar(cmap1, ax=axs[0])
        cbar1.set_label('eta (m)')
        axs[0].set_xlabel('X (m)')
        axs[0].set_ylabel('Y (m)', fontsize=14)
        axs[0].set_xlim(x_limits)
        axs[0].set_ylim(y_limits)


        # X, Y, UV plot with quivers
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
        axs[1].quiver(X_sampled, Y_sampled, U_test_sampled, V_test_sampled, color='black', scale=scale)
        axs[1].quiver(X_sampled, Y_sampled, U_pred_sampled, V_pred_sampled, color='red', scale=scale)
        axs[1].set_xlabel('X (m)', fontsize=14)
        axs[1].set_xlim(x_limits)
        axs[1].set_ylim(y_limits)


        # X, Y, h plot
        cmap3 = axs[2].pcolor(X_test, Y_test, h_pred_reshaped, shading='auto')
        cbar3 = fig.colorbar(cmap3, ax=axs[2])
        cbar3.set_label('bathymetry (m)')
        axs[2].set_xlabel('X (m)', fontsize=14)
        axs[2].set_xlim(x_limits)
        axs[2].set_ylim(y_limits)
        
        # Save the plot with file number in the filename
        plt.savefig(f'../plots/predictions_{file_suffix}.png')

        plt.tight_layout()
        plt.show()
        plt.close()  # Close the plot to free up memory
        
        # Print the current value of t
        print(f'Figure for t = {t} is plotted')
        
        # Concatenate the predictions for saving
        # predictions = np.hstack([h_pred, z_pred, u_pred, v_pred])

        # Save to a file
        # np.savetxt(f'../pinn_data/predictions_{file_suffix}.txt', predictions, delimiter=',', header='h_pred,z_pred,u_pred', comments='')
