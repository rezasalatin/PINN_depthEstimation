import torch
import numpy as np
import json
import os
from physics import Boussinesq_simple as physics_loss_calculator

class PINN:
    def __init__(self, model_path, config):
        self.config = config
        self.device = self.set_device()
        self.model_path = model_path
        self.model = self.load_model()
        self.init_optimizers()

        self.test_input_vars = config['data_residual']['inputs']
        self.test_output_vars = config['data_residual']['outputs']

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
            exit(1)

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
        test_input_data = np.column_stack([test_input_data[key].flatten() for key in self.test_input_vars])
        test_input_data = torch.tensor(test_input_data).float().to(self.device)

        initial_predictions = self.model(test_input_data)

        for i, var_name in enumerate(self.test_input_vars):
            setattr(self, var_name, test_input_data[:, i:i+1])

        for i, var_name in enumerate(self.test_output_vars):
            setattr(self, var_name, initial_predictions[:, i:i+1])

        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = physics_loss_calculator(self.t, self.x, self.y, self.h, self.z, self.u, self.v, initial_predictions)
            if loss.requires_grad:
                loss.backward()
            return loss

        self.optimizer_LBFGS.step(closure)

        with torch.no_grad():
            optimized_predictions = self.model(test_input_data)

        optimized_predictions = optimized_predictions.cpu().numpy()

        return optimized_predictions

if __name__ == "__main__":
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        exit(1)

    model_path = os.path.join(config['test']['model_path'], 'model.pth')

    folder = config['numerical_model']['dir']
    x_min, x_max = config['numerical_model']['x_min'], config['numerical_model']['x_max']
    y_min, y_max = config['numerical_model']['y_min'], config['numerical_model']['y_max']
    dt = config['numerical_model']['dt']

    file_no = config['numerical_model']['number_of_files']

    tester = PINN(model_path, config)

    x = np.linspace(x_min, x_max, num=config['numerical_model']['nx']).astype(np.float64)
    y = np.linspace(y_min, y_max, num=config['numerical_model']['ny']).astype(np.float64)
    X_test, Y_test = np.meshgrid(x, y)

    for i in range(file_no):
        test_input_data = {'x': X_test, 'y': Y_test, 't': np.full(X_test.shape, i*dt)}

        for key, value in config['data_residual']['inputs'].items():
            if key not in ['x', 'y', 't']:
                file_name = value["file"].format(i)
                file_path = os.path.join(folder, file_name)
                try:
                    data = np.loadtxt(file_path)
                    test_input_data[key] = data
                except Exception as e:
                    print(f"Error loading data from {file_path}: {e}")
                    continue

        test_outputs = tester.test(test_input_data)