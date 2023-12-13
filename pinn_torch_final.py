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
    def __init__(self, X_train, U_train, X_all, layers, X_star_min, X_star_max):
        
        # data
        self.t_u = torch.tensor(X_train[:, 0:1]).float().to(device)
        self.x_u = torch.tensor(X_train[:, 1:2]).float().to(device)
        self.y_u = torch.tensor(X_train[:, 2:3]).float().to(device)

        self.vals_min = torch.tensor(X_star_min).float().to(device)
        self.vals_max = torch.tensor(X_star_max).float().to(device)
        
        self.h_u = torch.tensor(U_train[:, 0:1]).float().to(device)
        self.z_u = torch.tensor(U_train[:, 1:2]).float().to(device)
        self.u_u = torch.tensor(U_train[:, 2:3]).float().to(device)
        self.v_u = torch.tensor(U_train[:, 3:4]).float().to(device)
        
        self.t_f = torch.tensor(X_all[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_all[:, 1:2], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_all[:, 2:3], requires_grad=True).float().to(device)
                
        self.layers = layers
        
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
            tolerance_grad=1e-7, 
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )

        # Define learning rate scheduler for Adam
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size=10000, 
            gamma=0.8
        )
        
    def net_u(self, t, x, y):  

        # input normalization between -1 and 1
        t = 2.0 * (t - self.vals_min[0])/(self.vals_max[0]-self.vals_min[0]) - 1.0
        x = 2.0 * (x - self.vals_min[1])/(self.vals_max[1]-self.vals_min[1]) - 1.0
        #y = 2.0 * (y - self.vals_min[2])/(self.vals_max[2]-self.vals_min[2]) - 1.0
        
        hzuv = self.dnn(torch.cat([t, x, y], dim=1))
        return hzuv
    
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
        
        hzuv_u_pred = self.net_u(self.t_u, self.x_u, self.y_u)
        
        h_pred = hzuv_u_pred[:, 0:1].to(device)
        z_pred = hzuv_u_pred[:, 1:2].to(device)
        u_pred = hzuv_u_pred[:, 2:3].to(device)
        v_pred = hzuv_u_pred[:, 3:4].to(device)
                
        loss_u = torch.mean((self.h_u - h_pred)**2) + \
            torch.mean((self.z_u - z_pred)**2) + \
            torch.mean((self.u_u - u_pred)**2) + \
            torch.mean((self.v_u - v_pred)**2)

        # Physics 
        hzuv_f_pred = self.net_u(self.t_f, self.x_f, self.y_f)

        h_pred = hzuv_f_pred[:, 0:1].to(device)
        z_pred = hzuv_f_pred[:, 1:2].to(device)
        u_pred = hzuv_f_pred[:, 2:3].to(device)
        v_pred = hzuv_f_pred[:, 3:4].to(device)
            
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
        
        weight_u = 1.0
        weight_f = 1.0
        
        loss = weight_u * loss_u + weight_f * loss_f
                
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss_u: %.5e, Loss_f: %.5e, Total Loss: %.5e' % (self.iter, loss_u.item(), loss_f.item(), loss.item())
            )

        return loss
    
    def train(self):
        self.dnn.train()

        # First phase of training with Adam
        for i in range(1000):  # 50,000 iterations
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
    
    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        hzuv_pred = self.net_u(t, x, y)
        
        h_pred = hzuv_pred[:, 0:1].to(device)
        z_pred = hzuv_pred[:, 1:2].to(device)
        u_pred = hzuv_pred[:, 2:3].to(device)
        v_pred = hzuv_pred[:, 3:4].to(device)
        return h_pred, z_pred, u_pred, v_pred  # Return the computed predictions
    
if __name__ == "__main__": 
    
    # Define some parameters
    Ntrain = 1000
    layers = [3, 30, 30, 30, 30, 30, 4] # layers
    # Extract all data.
    data = np.genfromtxt('../data/beach_2d.csv', delimiter=' ').astype(np.float32) # load data
    t_all = data[:, 0:1].astype(np.float64)
    x_all = data[:, 1:2].astype(np.float64)
    y_all = data[:, 2:3].astype(np.float64)
    h_all = data[:, 3:4].astype(np.float64)
    z_all = data[:, 4:5].astype(np.float64)
    u_all = data[:, 5:6].astype(np.float64)
    v_all = data[:, 6:7].astype(np.float64)
    
    X_star = np.hstack((t_all, x_all, y_all))
    # get data range for normalization
    X_star_min = np.min(X_star, axis=0)
    X_star_max = np.max(X_star, axis=0)

    U_star = np.hstack((h_all, z_all, u_all, v_all))
    
    idx = np.random.choice(X_star.shape[0], Ntrain, replace=False)
    
    # make a 1d list of data we have
    X_star_train = X_star[idx, :]       # inputs (t,x,y)
    U_star_train = U_star[idx, :]       # exact outputs (h,z,u,v)
    
    model = PhysicsInformedNN(X_star_train, U_star_train, X_star, layers, X_star_min, X_star_max)
    model.init_optimizers()  # Initialize optimizers
    # Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the results
    torch.save(model.dnn.state_dict(), './log/model.ckpt')
    # Testing
    X_test = np.arange(1024).reshape(-1, 1).astype(np.float64)  # Ensure correct shape
    Y_test = np.full((1024, 1), 1).astype(np.float64)           # Ensure correct shape
    T_test = np.full((1024, 1), 200).astype(np.float64)         # Ensure correct shape
    X_star = np.hstack((T_test, X_test, Y_test))  # Order: t, x, y
    h_pred, z_pred, u_pred, v_pred = model.predict(X_star)      
               
    # Concatenate the predictions for saving
    predictions = np.hstack([h_pred.detach().cpu().numpy(), 
                            z_pred.detach().cpu().numpy(), 
                            u_pred.detach().cpu().numpy(), 
                            v_pred.detach().cpu().numpy()])

    # Save to a file
    np.savetxt('predictions.txt', predictions, delimiter=',', header='h_pred,z_pred,u_pred,v_pred', comments='')         
               

    
