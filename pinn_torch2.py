"""
PINN with torch v2
@author: Reza Salatin
December 2023
w/ Pytorch
"""

import sys
sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict
import numpy as np
import time

np.random.seed(1234)

# CUDA support 
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    device = torch.device('cuda')
else:
    torch.manual_seed(1234)
    device = torch.device('cpu')
    
print(device)
    
    
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
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
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
    def __init__(self, X_train, U_train, X_all, layers):
        
        # data
        self.t = torch.tensor(X_train[:, 0:1], requires_grad=True).float().to(device)
        self.x = torch.tensor(X_train[:, 1:2], requires_grad=True).float().to(device)
        self.y = torch.tensor(X_train[:, 2:3], requires_grad=True).float().to(device)
        
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

        self.iter = 0

    # Initialize two optimizers
    def init_optimizers(self):
        # Adam optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), 
            lr=0.001  # You can adjust this learning rate
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
        
    def net_u(self, t, x, y):  
        hzuv = self.dnn(torch.cat([t, x, y], dim=1))
        return hzuv
    
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
            
        u_t = torch.autograd.grad(
            u_pred, self.t, 
            grad_outputs=torch.ones_like(u_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u_pred, self.x, 
            grad_outputs=torch.ones_like(u_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u_pred, self.y, 
            grad_outputs=torch.ones_like(u_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        
        v_t = torch.autograd.grad(
            v_pred, self.t,
            grad_outputs=torch.ones_like(v_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        v_x = torch.autograd.grad(
            v_pred, self.x,
            grad_outputs=torch.ones_like(v_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v_pred, self.y,
            grad_outputs=torch.ones_like(v_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        
        z_t = torch.autograd.grad(
            z_pred, self.t,
            grad_outputs=torch.ones_like(z_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        z_x = torch.autograd.grad(
            z_pred, self.x,
            grad_outputs=torch.ones_like(z_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        z_y = torch.autograd.grad(
            z_pred, self.y,
            grad_outputs=torch.ones_like(z_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # loss with physics (Navier Stokes / Boussinesq etc)
        f_u = u_t + (u_pred * u_x + v_pred * u_y) + 9.81 * (z_x + z_y)
        f_v = v_t + (u_pred * v_x + v_pred * v_y) + 9.81 * (z_x + z_y)
        f_c = z_t + (u_pred * z_x + v_pred * z_y) + (h_pred + z_pred) * (u_x + v_y)
        loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)
        
        loss = loss_u + loss_f
                
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss
    
    def train(self):
        self.dnn.train()
                
        # First phase of training with Adam
        for i in range(50000):  # 50,000 iterations
            self.optimizer_Adam.zero_grad()  # Zero gradients for Adam optimizer
            loss = self.loss_func()
            loss.backward()
            self.optimizer_Adam.step()

            #if i % 100 == 0:
            #    print('Adam Iter %d, Loss: %.5e' % (i, loss.item()))

        # Second phase of training with LBFGS
        def closure():
            self.optimizer_LBFGS.zero_grad()  # Zero gradients for LBFGS optimizer
            loss = self.loss_func()
            loss.backward()
            return loss

        for i in range(50000):  # Another 50,000 iterations
            self.optimizer_LBFGS.step(closure)

            #if i % 100 == 0:
            #    loss = closure()  # Recalculate to get current loss
            #    print('LBFGS Iter %d, Loss: %.5e' % (i, loss.item()))
    
    
if __name__ == "__main__": 
    
    # Define some parameters
    nIter = 50000   # iterations for training
    Ntrain = 4000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 4] # layers
    # Extract all data.
    data = np.genfromtxt('../../data/beach_2d.csv', delimiter=' ').astype(np.float32) # load data
    t_all = data[:, 0:1].astype(np.float64)
    x_all = data[:, 1:2].astype(np.float64)
    y_all = data[:, 2:3].astype(np.float64)
    h_all = data[:, 3:4].astype(np.float64)
    z_all = data[:, 4:5].astype(np.float64)
    u_all = data[:, 5:6].astype(np.float64)
    v_all = data[:, 6:7].astype(np.float64)
    
    X_star = np.hstack((t_all, x_all, y_all))
    U_star = np.hstack((h_all, z_all, u_all, v_all))
    
    idx = np.random.choice(X_star.shape[0], Ntrain, replace=False)
    
    # make a 1d list of data we have
    X_star_train = X_star[idx, :]       # inputs (t,x,y)
    U_star_train = U_star[idx, :]       # exact outputs (h,z,u,v)
    
    model = PhysicsInformedNN(X_star_train, U_star_train, X_star, layers)
    model.init_optimizers()  # Initialize optimizers
    # Training
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the results
    #torch.save(model.state_dict(), '../log/model.ckpt')
    # Testing
    #model.test()             
               

    
