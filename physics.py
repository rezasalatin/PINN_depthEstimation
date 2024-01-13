import torch

###########################################################
# compute gradients for the pinn
def compute_gradient(pred, var):

    grad = torch.autograd.grad(
        pred, var, 
        grad_outputs=torch.ones_like(pred),
        retain_graph=True,
        create_graph=True
    )[0]

    return grad


###########################################################
def Navier_Stokes(t, x, y, h, z, u, v, U, V):
    
    u_t = compute_gradient(u, t)
    u_x = compute_gradient(u, x)
    u_y = compute_gradient(u, y)

    v_t = compute_gradient(v, t)
    v_x = compute_gradient(v, x)
    v_y = compute_gradient(v, y)

    z_t = compute_gradient(z, t)
    z_x = compute_gradient(z, x)
    z_y = compute_gradient(z, y)

    h_x = compute_gradient(h+z, x)
    h_y = compute_gradient(h+z, y)

    hu_x = compute_gradient((h+z)*u, x)
    hv_y = compute_gradient((h+z)*v, y)

    # friction forces
    Fs_x = Cd*abs(U)*u
    Fs_y = Cd*abs(V)*v
    # breaking forces
    g = 9.81
    gamma_b = 0.78
    Fbr_x = 3.0/16.0 * g * gamma_b^2 * h_x * (h + z)
    Fbr_y = 3.0/16.0 * g * gamma_b^2 * h_y * (h + z)

    # loss with physics (Navier Stokes / Boussinesq etc)
    fc = z_t + hu_x + hv_y                                      # continuity eq
    fm_x = u_t + u * u_x + v * u_y + 9.81 * z_x + Fs_x + Fbr_x  # momentum eq in X dir
    fm_y = v_t + u * v_x + v * v_y + 9.81 * z_y + Fs_y + Fbr_y  # momentum eq in Y dir

    # loss
    loss = torch.mean(fc**2) + torch.mean(fm_x**2) + torch.mean(fm_y**2)

    return loss