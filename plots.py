import matplotlib.pyplot as plt
import os

###########################################################
# compute gradients for the pinn
def quiver(t, x, y, u1, v1, u2, v2, config):
    
    n = 10  # Interval: Sample every nth point (e.g., n=10 for every 10th grid point)
    scale = 25  # Arrow size: Adjust as needed for visibility
    axis_x, axis_y = x[::n, ::n], y[::n, ::n]
    u1, v1 = u1[::n, ::n], v1[::n, ::n]
    u2, v2 = u2[::n, ::n], v2[::n, ::n]

    # for all figures
    font_size = config['plot']['font_size']
    x_limits = config['plot']['x_limits']
    y_limits = config['plot']['y_limits']
    plot_folder = config['plot']['y_limits']

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