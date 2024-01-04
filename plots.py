import matplotlib.pyplot as plt
import os

###########################################################
def plot_quiver(t, x, y, u1, v1, u2, v2, config):
    
    n = 10  # Interval: Sample every nth point (e.g., n=10 for every 10th grid point)
    scale = 25  # Arrow size: Adjust as needed for visibility
    t = t[0, 0]
    axis_x, axis_y = x[::n, ::n], y[::n, ::n]
    u1, v1 = u1[::n, ::n], v1[::n, ::n]
    u2, v2 = u2[::n, ::n], v2[::n, ::n]
    
    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    y_limits = config['plot']['y_limits']
    plot_folder = config['plot']['dir']

    file_suffix = str(t).zfill(4)

    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
    ax.quiver(axis_x, axis_y, u1, v1, color='black', alpha=0.5, scale=scale)
    ax.quiver(axis_x, axis_y, u2, v2, color='red', alpha=0.5, scale=scale)
    ax.set_xlabel('X (m)', fontsize=font_size)
    ax.set_ylabel('Y (m)', fontsize=font_size)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    # Save the plot with file number in the filename
    # Ensure the directory exists
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, f'quiver_{file_suffix}sec.png'), dpi=300)
    plt.tight_layout()
    plt.close()
    
    
###########################################################
def plot_cmap(t, x, y, variable, config, var_name, v_min, v_max):
    
    t = t[0, 0]
    axis_x, axis_y = x, y
    axis_z = variable
    
    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    y_limits = config['plot']['y_limits']
    plot_folder = config['plot']['dir']

    file_suffix = str(t).zfill(4)
    
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
    
    cmap1 = ax.pcolor(axis_x, axis_y, axis_z, shading='auto', vmin=v_min, vmax=v_max)
    cbar1 = fig.colorbar(cmap1, ax=ax)
    cbar1.set_label(f'{var_name} (m)')
    ax.set_xlabel('X (m)', fontsize=font_size)
    ax.set_ylabel('Y (m)', fontsize=font_size)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    # Save the plot with file number in the filename
    # Ensure the directory exists
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, f'{var_name}_{file_suffix}sec.png'), dpi=300)
    plt.tight_layout()
    plt.close()
    
###########################################################
def plot_cmap_2column(t, x, y, variable_true, variable_pred, config, var_name, v_min, v_max):
    
    t = t[0, 0]
    axis_x, axis_y = x, y
    axis_z_pred = variable_pred
    axis_z_true = variable_true

    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    y_limits = config['plot']['y_limits']
    plot_folder = config['plot']['dir']

    file_suffix = str(t).zfill(4)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots side by side
    
    # First subplot for 'variable'
    cmap1 = axs[0].pcolor(axis_x, axis_y, axis_z_true, shading='auto', vmin=v_min, vmax=v_max)
    cbar1 = fig.colorbar(cmap1, ax=axs[0])
    #cbar1.set_label(f'{var_name} (m)')
    axs[0].set_xlabel('X (m)', fontsize=font_size)
    axs[0].set_ylabel('Y (m)', fontsize=font_size)
    axs[0].set_xlim(x_limits)
    axs[0].set_ylim(y_limits)

    # Second subplot for 'variable_true'
    cmap2 = axs[1].pcolor(axis_x, axis_y, axis_z_pred, shading='auto', vmin=v_min, vmax=v_max)
    cbar2 = fig.colorbar(cmap2, ax=axs[1])
    cbar2.set_label(f'{var_name}_true (m)')
    axs[1].set_xlabel('X (m)', fontsize=font_size)
    #axs[1].set_ylabel('Y (m)', fontsize=font_size)
    axs[1].set_xlim(x_limits)
    axs[1].set_ylim(y_limits)

    # Save the plot with file number in the filename
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, f'{var_name}_{file_suffix}sec.png'), dpi=300, bbox_inches='tight')
    plt.close()

###########################################################
def plot_2lines(t, x, y, variable_true, variable_pred, config, var_name, v_min, v_max):
    
    t = t[0]
    y = y[0]
    axis_x = x
    axis_y_pred = variable_pred
    axis_y_true = variable_true

    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    plot_folder = config['plot']['dir']

    file_suffix = str(t).zfill(4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(axis_x, axis_y_true, label='exact')
    plt.plot(axis_x, axis_y_pred, label='Predicted')
    plt.xlabel('Cross-Shore (m)')
    plt.ylabel(f'{var_name}, m')
    plt.xlim([x_limits[0], x_limits[1]])
    plt.ylim([v_min, v_max])
    plt.title(f'{var_name}')
    plt.legend()
    
    # Save the plot with file number in the filename
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, f'{var_name}_CS_{file_suffix}sec.png'), dpi=300, bbox_inches='tight')
    plt.close()

###########################################################
def plot_log(log_path, plot_path):
    
    # Initialize lists to store the extracted data
    iterations = []
    fidelity_losses = []
    residual_losses = []
    total_losses = []

    # Read the log file
    with open(os.path.join(log_path, 'log.txt'), 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            data = line.split(', ')
            iterations.append(int(data[0]))
            fidelity_losses.append(float(data[1]))
            residual_losses.append(float(data[2]))
            total_losses.append(float(data[3]))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, fidelity_losses, label='Fidelity Loss')
    plt.plot(iterations, residual_losses, label='Residual Loss')
    plt.plot(iterations, total_losses, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.xlim([0, 50000])
    plt.ylim([0.001, 10])
    plt.title('Loss Values Over Iterations')
    plt.yscale('log')
    plt.legend()
    
    # Save the plot with file number in the filename
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, 'log.png'), dpi=300, bbox_inches='tight')
    plt.close()
