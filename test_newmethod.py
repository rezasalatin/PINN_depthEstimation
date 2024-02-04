import torch
import numpy as np
from scipy.io import loadmat
import json
import sys
from physics import Navier_Stokes as physics_loss_calculator
import plots
import operations as op

# the physics-guided neural network
class pinn:
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = self.set_device()
        self.model_path = model_path
        self.model = self.load_model()
        self.init_optimizers()

        self.test_input_vars = config['data_test']['inputs']
        self.test_output_vars = config['data_test']['outputs']
        
        self.nx = config['data_test']['nx']
        self.ny = config['data_test']['ny']

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

    def test(self, test_input_data):
        test_input_data = torch.tensor(test_input_data).float().to(self.device)
        
        # temporal and spatial information for physics part
        for i, (key, info) in enumerate(self.test_input_vars.items()):
            tensor = test_input_data[:, i:i+1].clone().detach()
            if "true" in info["requires_grad"]:
                tensor = tensor.requires_grad_()
            tensor = tensor.float().to(self.device)
            setattr(self, key, tensor)
            
        test_input_data = [getattr(self, key) for i, key in enumerate(self.test_input_vars)]

        test_prediction_data = self.model(torch.cat(test_input_data, dim=-1))
        test_prediction_data = test_prediction_data.detach().cpu().numpy()

        return

if __name__ == "__main__":
    
    try:
        with open('config_CMB_h.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    model_path = config['data_test']['model']
    tester = pinn(model_path, config)

    inputs = config['data_test']['inputs']
    outputs = config['data_test']['outputs']

    file = config['data_test']['file']

    # Dictionary to store the loaded data
    test_input_data = np.empty((0, 1))
    test_input_dict = {}
    test_true_dict = {}

    for key in inputs:
        data = loadmat(file, variable_names=key)
        test_input_dict[key] = data[key]
        del data
        input_min_max = op.get_min_max(test_input_dict, key, config)
        test_input_dict[key] = op.normalize(test_input_dict[key], input_min_max[key][0], input_min_max[key][1])
    
        
    test_input = np.empty((0, 1))
    test_input_temp, test_input_flat = {}, {}
    
    for key in inputs:
        test_input_temp[key] = test_input_dict[key][:,:]
        # Flatten and reshape to ensure it's a column vector
        test_input_flat[key] = test_input_temp[key].reshape(-1, 1)
        # Concatenate the new array
        if test_input.size == 0:
            test_input = test_input_flat[key]
        else:
            test_input = np.hstack((test_input, test_input_flat[key]))   
    
    tester.test(test_input)
    
    print('Prediction done!')


