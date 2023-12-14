"""
PINN with torch v2
@author: Reza Salatin
December 2023
w/ Pytorch
"""

import torch
from collections import OrderedDict
import numpy as np
import time

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
    def __init__(self, X_u, X_f, layers, X_min, X_max):
        
        # data
        self.t_u = torch.tensor(X_u[:, 0:1]).float().to(device)
        self.x_u = torch.tensor(X_u[:, 1:2]).float().to(device)

        self.h_u = torch.tensor(X_u[:, 3:4]).float().to(device)
        self.z_u = torch.tensor(X_u[:, 4:5]).float().to(device)
        self.u_u = torch.tensor(X_u[:, 5:6]).float().to(device)
        
        self.t_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)

        self.layers = layers

        self.vals_min = torch.tensor(X_min).float().to(device)
        self.vals_max = torch.tensor(X_max).float().to(device)
        
        # deep neural networks
        self.dnn = DNN(layers).to(device)
        # initialize iteration number
        self.iter = 0

    # Initialize two optimizers
    def init_optimizers(self):
        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr=1e-4 # learning rate
        )

        # L-BFGS optimizer
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=62500, 
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
        
    def net_u(self, t, x):  

        # input normalization between -1 and 1
        t = 2.0 * (t - self.vals_min[0])/(self.vals_max[0]-self.vals_min[0]) - 1.0
        x = 2.0 * (x - self.vals_min[1])/(self.vals_max[1]-self.vals_min[1]) - 1.0
        
        output = self.dnn(torch.cat([t, x], dim=1))
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
        
        output_u_pred = self.net_u(self.t_u, self.x_u)
        
        h_pred = output_u_pred[:, 0:1].to(device)
        z_pred = output_u_pred[:, 1:2].to(device)
        u_pred = output_u_pred[:, 2:3].to(device)
        
        weight_h = 1
        weight_z = 100
        weight_u = 100
                
        loss_u = weight_h * torch.mean((self.h_u - h_pred)**2) + \
            weight_z * torch.mean((self.z_u - z_pred)**2) + \
            weight_u * torch.mean((self.u_u - u_pred)**2)
        
        # Physics 
        output_f_pred = self.net_u(self.t_f, self.x_f)

        h_pred = output_f_pred[:, 0:1].to(device)
        z_pred = output_f_pred[:, 1:2].to(device)
        u_pred = output_f_pred[:, 2:3].to(device)
            
        u_t = self.compute_gradient(u_pred, self.t_f)
        u_x = self.compute_gradient(u_pred, self.x_f)

        z_t = self.compute_gradient(z_pred, self.t_f)
        z_x = self.compute_gradient(z_pred, self.x_f)

        # loss with physics (Navier Stokes / Boussinesq etc)
        f_1 = u_t + u_pred * u_x + 9.81 * z_x
        f_2 = z_t + u_pred * z_x + (h_pred + z_pred) * u_x
        
        loss_f = torch.mean(f_1**2) + torch.mean(f_2**2)
        
        weight_loss_u = 1.0
        weight_loss_f = 1.0
        
        loss = weight_loss_u * loss_u + weight_loss_f * loss_f
                
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % (self.iter, loss_u.item(), loss_f.item(), loss.item())
            )

        return loss
    
    def train(self):
        self.dnn.train()

        # First phase of training with Adam
        for i in range(1000):  # number of iterations
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
        
        # testing data normalization between -1 and 1
        t = 2.0 * (t - self.vals_min[0])/(self.vals_max[0]-self.vals_min[0]) - 1.0
        x = 2.0 * (x - self.vals_min[1])/(self.vals_max[1]-self.vals_min[1]) - 1.0

        output_pred = self.net_u(t, x)
        
        h_pred = output_pred[:, 0:1].to(device)
        z_pred = output_pred[:, 1:2].to(device)
        u_pred = output_pred[:, 2:3].to(device)
        return h_pred, z_pred, u_pred  # Return the computed predictions
    
if __name__ == "__main__": 
    
    # Define some parameters
    Ntrain = 20000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3] # layers
    # Extract all data.
    data = np.genfromtxt('../data/beach_1d_dt001.csv', delimiter=' ').astype(np.float32) # load data
    t_all = data[:, 0:1].astype(np.float64)
    x_all = data[:, 1:2].astype(np.float64)
    y_all = data[:, 2:3].astype(np.float64)
    h_all = data[:, 3:4].astype(np.float64)
    z_all = data[:, 4:5].astype(np.float64)
    u_all = data[:, 5:6].astype(np.float64)
    v_all = data[:, 6:7].astype(np.float64)
    
    X_star = np.hstack((t_all, x_all, y_all, h_all, z_all, u_all, v_all))
    # get data range for normalization
    X_star_min = np.min(X_star, axis=0)
    X_star_max = np.max(X_star, axis=0)
    
    idx = np.random.choice(X_star.shape[0], Ntrain, replace=False)
    
    # make a 1d list of data we have
    X_u_star = X_star[idx,:]       # inputs (t,x,y)
    X_f_star = X_star               # inputs for residuals (t,x,y)
    
    model = PhysicsInformedNN(X_u_star, X_f_star, layers, X_star_min, X_star_max)
    model.init_optimizers()  # Initialize optimizers
    # Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the results
    torch.save(model.dnn.state_dict(), './log/model.ckpt')
    # Testing
    T_test = np.full((1024, 1), 300).astype(np.float64)         # Ensure correct shape
    X_test = np.arange(1024).reshape(-1, 1).astype(np.float64)  # Ensure correct shape
    X_star = np.hstack((T_test, X_test))  # Order: t, x, y
    h_pred, z_pred, u_pred = model.predict(X_star)

    # Concatenate the predictions for saving
    predictions = np.hstack([h_pred.detach().cpu().numpy(), 
                            z_pred.detach().cpu().numpy(), 
                            u_pred.detach().cpu().numpy()])

    # Save to a file
    np.savetxt('../data/predictions.txt', predictions, delimiter=',', header='h_pred,z_pred,u_pred', comments='')