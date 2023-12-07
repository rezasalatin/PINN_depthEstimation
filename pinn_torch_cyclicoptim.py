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
        self.t = torch.tensor(X_train[:, 0:1], requires_grad=True).float().to(device)
        self.x = torch.tensor(X_train[:, 1:2], requires_grad=True).float().to(device)
        self.y = torch.tensor(X_train[:, 2:3], requires_grad=True).float().to(device)

        self.vals_min = torch.tensor(X_star_min).float().to(device)
        self.vals_max = torch.tensor(X_star_max).float().to(device)
        
        self.h = torch.tensor(U_train[:, 0:1]).float().to(device)
        self.z = torch.tensor(U_train[:, 1:2]).float().to(device)
        self.u = torch.tensor(U_train[:, 2:3]).float().to(device)
        self.v = torch.tensor(U_train[:, 3:4]).float().to(device)
        
        self.t_f = torch.tensor(X_all[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_all[:, 1:2], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_all[:, 2:3], requires_grad=True).float().to(device)
                
        self.layers = layers
        
        # deep neural networks
        self.dnn = DNN(layers).to(device)
        # initialize iteration number
        self.iter = 0
        # Add a counter for LBFGS iterations
        self.lbfgs_iter = 0

        # initialize dynamic weights
        self.alpha = 1.0  # initial value for alpha
        self.lambda_val = 0.1 # initialize lambda

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
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-10, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        # Define learning rate scheduler for Adam
        self.scheduler_Adam = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, 
            step_size=5000, 
            gamma=0.8
        )
        
    def net_u(self, t, x, y):  

        # input normalization between -1 and 1
        t_norm = 2.0 * (t - self.vals_min[0])/(self.vals_max[0]-self.vals_min[0]) - 1.0
        x_norm = 2.0 * (x - self.vals_min[1])/(self.vals_max[1]-self.vals_min[1]) - 1.0
        y_norm = y #2.0 * (y - self.vals_min[2])/(self.vals_max[2]-self.vals_min[2]) - 1.0
        
        hzuv = self.dnn(torch.cat([t_norm, x_norm, y_norm], dim=1))
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
    
    # dynamic wieghts
    def compute_dynamic_weights(self):
        # Calculate gradients
        gradients_lf = torch.autograd.grad(self.loss_lf, self.dnn.parameters(), retain_graph=True)
        gradients_lu = torch.autograd.grad(self.loss_lu, self.dnn.parameters(), retain_graph=True)

        # Compute max of gradients for Le and mean for Lb and Li
        gradients_lf = torch.mean(torch.stack([torch.norm(g, 1) for g in gradients_lf]))
        gradients_lu = max([torch.norm(g, 1) for g in gradients_lu])

        # Update alpha and beta using the dynamic weight strategy
        self.alpha = (1 - self.lambda_val) * self.alpha \
            + self.lambda_val * (gradients_lu / gradients_lf)
    
    def loss_func(self):
        
        hzuv_pred = self.net_u(self.t, self.x, self.y)
        
        h_pred = hzuv_pred[:, 0:1].to(device)
        z_pred = hzuv_pred[:, 1:2].to(device)
        u_pred = hzuv_pred[:, 2:3].to(device)
        v_pred = hzuv_pred[:, 3:4].to(device)
                
        loss_u = torch.mean((self.h - h_pred)**2) + \
            torch.mean((self.z - z_pred)**2) + \
            torch.mean((self.u - u_pred)**2) + \
            torch.mean((self.v - v_pred)**2)
            
        u_t = self.compute_gradient(u_pred, self.t)
        u_x = self.compute_gradient(u_pred, self.x)
        u_y = self.compute_gradient(u_pred, self.y)

        v_t = self.compute_gradient(v_pred, self.t)
        v_x = self.compute_gradient(v_pred, self.x)
        v_y = self.compute_gradient(v_pred, self.y)

        z_t = self.compute_gradient(z_pred, self.t)
        z_x = self.compute_gradient(z_pred, self.x)
        z_y = self.compute_gradient(z_pred, self.y)

        hu_x = self.compute_gradient(h_pred * u_pred, self.x)
        hv_y = self.compute_gradient(h_pred * v_pred, self.y)

        f_c = z_t + (hu_x + hv_y)
        f_u = u_t + (u_pred * u_x + v_pred * u_y) + 9.81 * z_x
        f_v = v_t + (u_pred * v_x + v_pred * v_y) + 9.81 * z_y
        
        loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)
        loss_f = self.alpha * loss_f

        loss = loss_f + loss_u

        # Store individual loss components for dynamic weight calculation
        self.loss_lf = loss_f
        self.loss_lu = loss_u
         
        self.iter += 1
        if self.iter % 100 == 0:
            # Retrieve the current learning rate for Adam optimizer
            current_lr = self.optimizer_Adam.param_groups[0]['lr']
            
            print(
            'Iter %d, LR: %.2e, Alpha: %.2e, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e' % 
            (self.iter, current_lr, self.alpha, loss.item(), loss_u.item(), loss_f.item())
            )   
        return loss
    
    def train(self, num_cycles, adam_iters, lbfgs_iters):
        self.dnn.train()

        best_loss = float('inf')
        best_model_state = None

        for _ in range(num_cycles):
            # Adam phase
            for i in range(adam_iters):
                self.optimizer_Adam.zero_grad()
                loss = self.loss_func()
                loss.backward()
                self.compute_dynamic_weights()
                self.optimizer_Adam.step()
                self.scheduler_Adam.step()

                # Track the best model and loss
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model_state = self.dnn.state_dict().copy()  # Save the best model state

                # Print the current status
                if self.iter % 100 == 0:
                    current_lr = self.optimizer_Adam.param_groups[0]['lr']
                    print('Adam Iter %d, LR: %.2e, Alpha: %.2e, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e' %
                          (self.iter, current_lr, self.alpha, loss.item(), loss_u.item(), loss_f.item()))
                    
            # Check if Adam improved the loss
            if loss.item() < best_loss:
                # Update the best loss and model state
                best_loss = loss.item()
                best_model_state = self.dnn.state_dict().copy()

            else:
                # Restore the best model state before starting LBFGS
                self.dnn.load_state_dict(best_model_state)
            
            # LBFGS phase
            def closure():
                self.optimizer_LBFGS.zero_grad()
                loss = self.loss_func()
                loss.backward()
                return loss

            for i in range(lbfgs_iters):
                self.optimizer_LBFGS.step(closure)

                # Track the best model and loss
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model_state = self.dnn.state_dict().copy()

                # Print the current status in LBFGS phase
                if self.lbfgs_iter % 100 == 0:
                    print('LBFGS Iter %d, Alpha: %.2e, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e' %
                          (self.lbfgs_iter, self.alpha, loss.item(), loss_u.item(), loss_f.item()))
                self.lbfgs_iter += 1

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
    Ntrain = 2000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 4] # layers

    # Specify the number of cycles and iterations for Adam and LBFGS
    NUM_CYCLES = 1000
    ADAM_ITERS = 50
    LBFGS_ITERS = 50

    # Extract all data.
    data = np.genfromtxt('../data/beach_1d.csv', delimiter=' ').astype(np.float32) # load data
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
    model.train(NUM_CYCLES, ADAM_ITERS, LBFGS_ITERS)
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