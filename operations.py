import numpy as np

# Normalization input data
def normalize(data, data_min, data_max):
    if data_max == data_min:
        return np.zeros_like(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# Find min and max for each variable for normalization
def get_min_max(data):
    min_max_dict = {}
    for key in data:
        min_max_dict[key] = (data[key].min(), data[key].max())
    return min_max_dict