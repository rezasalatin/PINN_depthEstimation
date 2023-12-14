import pandas as pd
import numpy as np

# t time, x hori, y vert, h depth, z surf elev, u vel in hori, v vel in vert

# Define the directory path
directory = r"../../../FUNWAVE-TVD/simple_cases/surface_wave_1d/input_files"

# List to store the selected data from each file
all_selected_rows = pd.DataFrame(columns=['t', 'x', 'y', 'h', 'z', 'u', 'v'])

#random_indices = np.random.choice(400, 100, replace=False)  # 100 unique indices from first 400 rows

# Loop through each file    
for i in range(1, 21):
    # t, z, u, v
    station_name = f"{directory}/output/sta_{i:04}"  # Adjust the file extension if needed
    # Read the file, assuming whitespace or tab delimited
    data = pd.read_csv(station_name, delim_whitespace=True, header=None, names=['t', 'z', 'u', 'v'])
    # Round the columns to the specified precision
    data['t'] = data['t'].round(2)   # Round first  column ('t') to 0.01 s
    data['z'] = data['z'].round(3)   # Round second column ('z') to 0.001 m
    data['u'] = data['u'].round(3)   # Round third  column ('u') to 0.001 m/s
    data['v'] = data['v'].round(3)   # Round fourth column ('v') to 0.001 m/s
    # Select the first 401 rows
    data = data.iloc[5092:10190]
    # Randomly select rows
    selected_rows = data #data.iloc[random_indices]
    
    # x, y
    gauge_file = f"{directory}/gauges.txt"
    # Read the gauges file and extract X and Y values
    gauges_data = pd.read_csv(gauge_file, delim_whitespace=True, header=None)
    idX, idY = gauges_data.iloc[i-1, :2]
    # Add X and Y values to each row
    selected_rows.insert(1, 'y', (idY-1)*1.0)
    selected_rows.insert(1, 'x', (idX-1)*1.0)

    # depth
    depth_file = f"{directory}/output/dep.out"
    # Read the depth file
    depth = pd.read_csv(depth_file, delim_whitespace=True, header=None)
    h = depth.iloc[idY-1, idX-1]
    h = h.round(2)   # Round h to 0.01
    selected_rows.insert(3, 'h', h)

    all_selected_rows = pd.concat([all_selected_rows, selected_rows], ignore_index=True)

final_data = all_selected_rows

# Sort the selected rows based on 't'
# final_data = final_data.sort_values(by=['t', 'x', 'y'])

# Define the name for the combined extracted file
extracted_file_name = f"{directory}/beach1d_2.csv"

# Save the combined data to the new file
final_data.to_csv(extracted_file_name, sep=' ', index=False, header=False)
