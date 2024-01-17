import numpy as np

# Normalize input data
def normalize(data, data_min, data_max):
    if data_max == data_min:
        return np.zeros_like(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# Denormalize input data
def denormalize(data, data_min, data_max):
    if data_max == data_min:
        return np.zeros_like(data_min)
    return (data + 1) / 2 * (data_max - data_min) + data_min

# Find min and max for each variable for normalization
def get_min_max(data, config):
    
    x_min, x_max = config['data_test']['x_min'], config['data_test']['x_max']
    y_min, y_max = config['data_test']['y_min'], config['data_test']['y_max']
    
    min_max_dict = {}
    for key in data:
        if key == 'x':
            min_max_dict[key] = (x_min, x_max)
        elif key == 'y':
            min_max_dict[key] = (y_min, y_max)
        else:
            min_value = np.nanmin(data[key])
            max_value = np.nanmax(data[key])
            min_max_dict[key] = (min_value, max_value)
    return min_max_dict
