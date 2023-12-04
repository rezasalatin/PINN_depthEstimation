"""
PINN
@author: Reza Salatin
December 2023
w/ Pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# random see for numpy
np.random.seed(1234)

# set device and random seeds
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    device = torch.device('cuda')
else:
    torch.manual_seed(1234)
    device = torch.device('cpu')

######################################################################
class PINN(nn.Module):

    ## initiate the class

    def __init__(self, layers):

        super(PINN, self).__init__()
        self.net_layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net_layers.append(nn.Tanh())
    
    ## forward 
    
    def forward(self, inputs):

        for layer in self.net_layers:
            inputs = layer(inputs)
        return inputs
    
    ## calculate loss with data and physics
    
    def loss_func(self, input_vals, true_vals, pred_vals):
        
        pred_vals = self(input_vals)
        t_in, x_in, y_in = input_vals[:, 0], input_vals[:, 1], input_vals[:, 2]
        h_true, z_true, u_true, v_true = true_vals[:, 0], true_vals[:, 1], true_vals[:, 2], true_vals[:, 3]
        h_pred, z_pred, u_pred, v_pred = pred_vals[:, 0], pred_vals[:, 1], pred_vals[:, 2], pred_vals[:, 3]

        # loss with data
        loss_data_fidelity = torch.mean((h_true - h_pred)**2) + \
            torch.mean((z_true - z_pred)**2) + \
            torch.mean((u_true - u_pred)**2) + \
            torch.mean((v_true - v_pred)**2)

        # gradients
        u_t = torch.autograd.grad(u_pred, t_in, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x_in, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred, y_in, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        v_t = torch.autograd.grad(v_pred, t_in, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        v_x = torch.autograd.grad(v_pred, x_in, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        v_y = torch.autograd.grad(v_pred, y_in, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        z_t = torch.autograd.grad(z_pred, t_in, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]
        z_x = torch.autograd.grad(z_pred, x_in, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]
        z_y = torch.autograd.grad(z_pred, y_in, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]

        # loss with physics
        f_u = u_t + (u_pred * u_x + v_pred * u_y) + 9.81 * (z_x + z_y)
        f_v = v_t + (u_pred * v_x + v_pred * v_y) + 9.81 * (z_x + z_y)
        f_c = z_t + (u_pred * z_x + v_pred * z_y) + (h_pred + z_pred) * (u_x + v_y)
        loss_physics_constraints = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)

        # Total loss
        loss = loss_data_fidelity + loss_physics_constraints

        return loss
    
    ## train the model
    
    def train(self, train_data, nIter):

        # With Adam optimizer
        optimizer_adam = optim.Adam(self.parameters(), lr=0.001)
        start_time = time.time()
        for it in range(nIter):
            if it % 100 == 0:
                print(it)
            for t_train, x_train, y_train, h_train, z_train, u_train, v_train in train_data:

                t_train, x_train, y_train, h_train, z_train, u_train, v_train = \
                t_train.to(device), x_train.to(device), y_train.to(device), \
                h_train.to(device), z_train.to(device), u_train.to(device), v_train.to(device)

                inputs = torch.cat((t_train, x_train, y_train), dim=1)
                true_vals = torch.cat((h_train, z_train, u_train, v_train), dim=1)

                optimizer_adam.zero_grad()
                loss = self.loss_func(inputs, true_vals)
                loss.backward()
                optimizer_adam.step()
                
                # Print loss every 10 iterations.
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss.item(), elapsed))
                    start_time = time.time()

        # Switch to L-BFGS optimizer / careful! this uses stale input data
        optimizer_lbfgs = optim.LBFGS(self.parameters(), lr=1)
        for i in range(nIter):
            for t_train, x_train, y_train, h_train, u_train, v_train, z_train in train_data:

                t_train, x_train, y_train, h_train, z_train, u_train, v_train = \
                t_train.to(device), x_train.to(device), y_train.to(device), \
                h_train.to(device), z_train.to(device), u_train.to(device), v_train.to(device)

                inputs = torch.cat((t_train, x_train, y_train), dim=1)
                true_vals = torch.cat((h_train, z_train, u_train, v_train), dim=1)
                
                optimizer_lbfgs.zero_grad()
                pred_vals = self(inputs)
                loss = self.loss_func(inputs, true_vals, pred_vals)
                loss.backward()
                optimizer_lbfgs.step()
                print(f'Iter {it}, Loss: {loss.item()}')            
   
    ## test the model
   
    def test(self, test_data):
        model.eval()
        with torch.no_grad():
            for t_test, x_test, y_test, h_test, u_test, v_test, z_test in test_data:

                t_test, x_test, y_test, h_test, z_test, u_test, v_test = \
                t_test.to(device), x_test.to(device), y_test.to(device), \
                h_test.to(device), z_test.to(device), u_test.to(device), v_test.to(device)
                inputs = torch.cat((t_test, x_test, y_test), dim=1)
                true_vals = torch.cat((h_test, z_test, u_test, v_test), dim=1)
                pred_vals = self(inputs)
                loss = self.loss_func(inputs, true_vals, pred_vals)

                print(f'Test, Loss: {loss.item()}')

###############################################################################
# Main

if __name__ == "__main__": 
    # Define some parameters
    nIter = 50000   # iterations for training
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 4] # layers
    # Extract all data.
    data = np.genfromtxt('../data/beach_2d.csv', delimiter=' ').astype(np.float32) # load data
    t_all = torch.tensor(data[:, 0:1], requires_grad=True, dtype=torch.float32, device=device)
    x_all = torch.tensor(data[:, 1:2], requires_grad=True, dtype=torch.float32, device=device)
    y_all = torch.tensor(data[:, 2:3], requires_grad=True, dtype=torch.float32, device=device)
    h_all = torch.tensor(data[:, 3:4], dtype=torch.float32, device=device)
    z_all = torch.tensor(data[:, 4:5], dtype=torch.float32, device=device)
    u_all = torch.tensor(data[:, 5:6], dtype=torch.float32, device=device)
    v_all = torch.tensor(data[:, 6:7], dtype=torch.float32, device=device)
    # Shuffle and split the data into train and test datasets
    dataset = torch.utils.data.TensorDataset(t_all, x_all, y_all, h_all, z_all, u_all, v_all)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Create an instance of the PINN class with the specified layers
    model = PINN(layers).to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Training
    start_time = time.time() 
    model.train(train_loader, nIter)
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)
    # Save the results
    torch.save(model.state_dict(), '../log/model.ckpt')
    # Testing
    model.test(test_loader)