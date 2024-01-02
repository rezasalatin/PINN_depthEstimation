import numpy as np

# Normalization input data
def normalize(data, data_min, data_max):
    if data_max == data_min:
        return np.zeros_like(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# Find min and max for each variable for normalization
def get_min_max(data, config):
    
    x_min, x_max = config['numerical_model']['x_min'], config['numerical_model']['x_max']
    y_min, y_max = config['numerical_model']['y_min'], config['numerical_model']['y_max']
    
    min_max_dict = {}
    for key in data:
        if key == 'x':
            min_max_dict[key] = (x_min, x_max)
        elif key == 'y':
            min_max_dict[key] = (y_min, y_max)
        else:
            min_max_dict[key] = (data[key].min(), data[key].max())
    return min_max_dict