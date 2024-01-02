import torch
import numpy as np
import json
import os
import sys
from physics import Boussinesq_simple as physics_loss_calculator
import plots

# the physics-guided neural network
class pinn:
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = self.set_device()
        self.model_path = model_path
        self.model = self.load_model()
        self.init_optimizers()

        self.test_input_vars = config['data_residual']['inputs']
        self.test_output_vars = config['data_residual']['outputs']
        
        self.nx = config['numerical_model']['nx']
        self.ny = config['numerical_model']['ny']

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Device in use: GPU")
        else:
            device = torch.device('cpu')
            print("Device in use: CPU")
        return device

    def load_model(self):
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def init_optimizers(self):
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(), 
            lr=self.config['lbfgs_optimizer']['learning_rate'],
            max_iter=1, 
            max_eval=2, 
            history_size=10,
            tolerance_grad=self.config['lbfgs_optimizer']['tolerance_grad'],
            tolerance_change=self.config['lbfgs_optimizer']['tolerance_change'],
            line_search_fn=self.config['lbfgs_optimizer']['line_search_fn']
        )

    def test(self, test_input_data, test_true_data, file_no):
        test_input_data = torch.tensor(test_input_data).float().to(self.device)
        
        # temporal and spatial information for physics part
        for i, (var_name, var_info) in enumerate(self.test_input_vars.items()):
            tensor = test_input_data[:, i:i+1].clone().detach()
            if "true" in var_info["requires_grad"]:
                tensor = tensor.requires_grad_()
            tensor = tensor.float().to(self.device)
            setattr(self, var_name, tensor)

            # Clone the tensor to a new variable with the prefix 'plot_'
            plot_tensor = tensor.clone().detach()  # Clone and detach the tensor
            plot_tensor = plot_tensor.reshape(self.ny, self.nx)
            setattr(self, f'plot_input_{var_name}', plot_tensor)

            
        test_input_data = [getattr(self, var_name) for i, var_name in enumerate(self.test_input_vars)]

        test_prediction_data = self.model(torch.cat(test_input_data, dim=-1))

        for i, var_name in enumerate(self.test_output_vars):
            tensor = test_prediction_data[:, i:i+1]
            setattr(self, var_name, tensor)
            # Clone the tensor to a new variable with the prefix
            plot_tensor = tensor.clone().detach()  # Clone and detach the tensor
            plot_tensor = plot_tensor.reshape(self.ny, self.nx)
            setattr(self, f'plot_pred_{var_name}', plot_tensor)
            
            # Convert the NumPy array to a PyTorch tensor, then clone and detach
            tensor = torch.from_numpy(test_true_data[var_name]).clone().detach()
            setattr(self, f'plot_true_{var_name}', tensor)

        # Check if optimization is to be performed
        if self.config.get('perform_optimization', False):

            def closure():
                self.optimizer_LBFGS.zero_grad()
                loss = physics_loss_calculator(self.t, self.x, self.y, self.h, self.z, self.u, self.v)
                if loss.requires_grad:
                    loss.backward()
                return loss

            self.optimizer_LBFGS.step(closure)

            with torch.no_grad():
                test_prediction_data = self.model(torch.cat(test_input_data, dim=-1))

        test_prediction_data = test_prediction_data.detach().cpu().numpy()

        # plots
        plots.plot_quiver(self.plot_input_t, self.plot_input_x, self.plot_input_y, self.plot_input_u, self.plot_input_v, self.plot_pred_u, self.plot_pred_v, self.config)
        
        #plots.plot_cmap(self.plot_input_t, self.plot_input_x, self.plot_input_y, self.plot_pred_z, self.config, 'eta')
        #plots.plot_cmap(self.plot_input_t, self.plot_input_x, self.plot_input_y, self.plot_pred_h, self.config, 'depth')
        
        plots.plot_cmap_2column(self.plot_input_t, self.plot_input_x, self.plot_input_y, self.plot_true_z, self.plot_pred_z, self.config, 'eta')
        plots.plot_cmap_2column(self.plot_input_t, self.plot_input_x, self.plot_input_y, self.plot_true_h, self.plot_pred_h, self.config, 'depth')

        return test_prediction_data

if __name__ == "__main__":
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    model_path = os.path.join(config['test']['model_path'], 'model.pth')

    folder = config['numerical_model']['dir']
    x_min, x_max = config['numerical_model']['x_min'], config['numerical_model']['x_max']
    y_min, y_max = config['numerical_model']['y_min'], config['numerical_model']['y_max']
    dt = config['numerical_model']['dt']

    num_files = config['numerical_model']['number_of_files']

    tester = pinn(model_path, config)

    x = np.linspace(x_min, x_max, num=config['numerical_model']['nx']).astype(np.float64)
    y = np.linspace(y_min, y_max, num=config['numerical_model']['ny']).astype(np.float64)
    X_test, Y_test = np.meshgrid(x, y)

    for file_no in range(200, num_files):
        
        file_suffix = str(file_no).zfill(5)

        # Dictionary to store the loaded data
        test_input_data = {}
        test_true_data = {}

        # Iterate over the mapping and load each file
        for key, value in config['data_residual']['inputs'].items():
            
            file_name = value["file"]
            if key == 'x':
                data = X_test
            elif key == 'y':
                data = Y_test
            elif key == 't':
                data = np.full(X_test.shape, file_no*dt, dtype=np.float64)
            else:
                fname = file_name if key == 'h' else f"{file_name}_{file_suffix}"    
                file_path = os.path.join(folder, fname)
                data = np.loadtxt(file_path)
                
            test_input_data[key] = data
            
        # Iterate over the mapping and load each file
        for key, value in config['data_residual']['outputs'].items():
            
            file_name = value["file"]
            fname = file_name if key == 'h' else f"{file_name}_{file_suffix}"    
            file_path = os.path.join(folder, fname)
            data = np.loadtxt(file_path)
                
            test_true_data[key] = data
            
        test_input_data = np.column_stack([test_input_data[key].flatten() for key in config['data_residual']['inputs']])

        test_outputs = tester.test(test_input_data, test_true_data, file_no)
        print(f'Done: Prediction for file: {file_no}')


