import matplotlib.pyplot as plt
import os

###########################################################
def plot_quiver(t, x, y, u1, v1, u2, v2, config):
    
    n = 10  # Interval: Sample every nth point (e.g., n=10 for every 10th grid point)
    scale = 25  # Arrow size: Adjust as needed for visibility
    t = t[0, 0].cpu().numpy()
    axis_x, axis_y = x[::n, ::n].cpu().numpy(), y[::n, ::n].cpu().numpy()
    u1, v1 = u1[::n, ::n].cpu().numpy(), v1[::n, ::n].cpu().numpy()
    u2, v2 = u2[::n, ::n].cpu().numpy(), v2[::n, ::n].cpu().numpy()
    
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
def plot_cmap(t, x, y, variable, config, var_name):
    
    t = t[0, 0].cpu().numpy()
    axis_x, axis_y = x.cpu().numpy(), y.cpu().numpy()
    axis_z = variable.cpu().numpy()
    
    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    y_limits = config['plot']['y_limits']
    plot_folder = config['plot']['dir']

    file_suffix = str(t).zfill(4)
    
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed
    
    cmap1 = ax.pcolor(axis_x, axis_y, axis_z, shading='auto')
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
def plot_cmap_2column(t, x, y, variable_true, variable_pred, config, var_name):
    
    t = t[0, 0].cpu().numpy()
    axis_x, axis_y = x.cpu().numpy(), y.cpu().numpy()
    axis_z_pred = variable_pred.cpu().numpy()
    axis_z_true = variable_true.cpu().numpy()

    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    y_limits = config['plot']['y_limits']
    plot_folder = config['plot']['dir']

    file_suffix = str(t).zfill(4)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots side by side
    
    # First subplot for 'variable'
    cmap1 = axs[0].pcolor(axis_x, axis_y, axis_z_true, shading='auto')
    cbar1 = fig.colorbar(cmap1, ax=axs[0])
    #cbar1.set_label(f'{var_name} (m)')
    axs[0].set_xlabel('X (m)', fontsize=font_size)
    axs[0].set_ylabel('Y (m)', fontsize=font_size)
    axs[0].set_xlim(x_limits)
    axs[0].set_ylim(y_limits)

    # Second subplot for 'variable_true'
    cmap2 = axs[1].pcolor(axis_x, axis_y, axis_z_pred, shading='auto')
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