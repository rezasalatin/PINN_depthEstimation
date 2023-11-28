import pandas as pd
import numpy as np

# Define the directory path
directory = r"\\wsl.localhost\Ubuntu-22.04\home\reza\projects\Boussinesq\FUNWAVE-TVD\simple_cases\beach_2d\input_files"

# List to store the selected data from each file
all_selected_rows = pd.DataFrame(columns=['t', 'x', 'y', 'eta', 'u', 'v'])

random_indices = np.random.choice(400, 100, replace=False)  # 100 unique indices

# Loop through each file
for i in range(1, 51):

    # t, eta, u, v
    station_name = f"{directory}\\output\sta_{i:04}"  # Adjust the file extension if needed
    # Read the file, assuming whitespace or tab delimited
    data = pd.read_csv(station_name, delim_whitespace=True, header=None, names=['t', 'eta', 'u', 'v'])
    # Round the columns to the specified precision
    data['t'] = data['t'].round(2)   # Round first column ('t') to 0.01
    data['eta'] = data['eta'].round(3)  # Round second column ('eta') to 0.001
    data['u'] = data['u'].round(3)   # Round third column ('u') to 0.001
    data['v'] = data['v'].round(3)   # Round fourth column ('v') to 0.001
    # Select the first 400 rows
    data = data.head(400)
    # Randomly select 100 rows
    selected_rows = data.iloc[random_indices]
    
    # x, y
    gauge_file = f"{directory}\\gauges.txt"  # Adjust the file extension if needed
    # Read the gauges file and extract X and Y values from the first row
    gauges_data = pd.read_csv(gauge_file, delim_whitespace=True, header=None)
    X, Y = gauges_data.iloc[i-1, :2]
    # Add X and Y values to each row
    selected_rows.insert(1, 'y', (Y-1)*2.0)
    selected_rows.insert(1, 'x', (X-1)*2.0)

    all_selected_rows = pd.concat([all_selected_rows, selected_rows], ignore_index=True)  # ignore_index resets row index

final_data = all_selected_rows

# Sort the selected rows based on 't'
final_data = final_data.sort_values(by=['t', 'x', 'y'])

# Define the name for the combined extracted file
extracted_file_name = f"{directory}\\extracted.csv"

# Save the combined data to the new file
final_data.to_csv(extracted_file_name, sep=' ', index=False, header=False)