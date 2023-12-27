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
def Boussinesq_simple(t, x, y, h, z, u, v, device):
    
    # This u is not correct. It is at a specific depth. For equations, calculate the u at the surface.

    u_t = compute_gradient(u, t)
    u_x = compute_gradient(u, x)
    u_y = compute_gradient(u, y)

    v_t = compute_gradient(v, t)
    v_x = compute_gradient(v, x)
    v_y = compute_gradient(v, y)

    z_t = compute_gradient(z, t)
    z_x = compute_gradient(z, x)
    z_y = compute_gradient(z, y)

    # Higher orders (refer to Shi et al. 2012, Ocean Modeling)
    hu = h * u
    hv = h * v
    hu_x = compute_gradient(hu, x)
    hv_y = compute_gradient(hv, y)

    # loss with physics (Navier Stokes / Boussinesq etc)
    f_cont = z_t + hu_x + hv_y # continuity eq.
    f_momx = u_t + u * u_x + v * u_y + 9.81 * z_x   # momentum in X dir
    f_momy = v_t + u * v_x + v * v_y + 9.81 * z_y   # momentum in Y dir
    
    loss_f = torch.mean(f_cont**2) + torch.mean(f_momx**2) + torch.mean(f_momy**2)

    return loss_f
    
    ###########################################################
def Boussinesq(output, t, x, y, device):

    h_pred = output[:, 0:1].to(device)
    z_pred = output[:, 1:2].to(device)
    u_pred = output[:, 2:3].to(device)
    v_pred = output[:, 3:4].to(device)

    # This u is not correct. It is at a specific depth. For equations, calculate the u at the surface.

    u_t = compute_gradient(u_pred, t)
    u_x = compute_gradient(u_pred, x)
    u_y = compute_gradient(u_pred, y)

    v_t = compute_gradient(v_pred, t)
    v_x = compute_gradient(v_pred, x)
    v_y = compute_gradient(v_pred, y)

    z_t = compute_gradient(z_pred, t)
    z_x = compute_gradient(z_pred, x)
    z_y = compute_gradient(z_pred, y)

    # Higher orders (refer to Shi et al. 2012, Ocean Modeling)
    hu = h_pred * u_pred
    hv = h_pred * v_pred
    hu_x = compute_gradient(hu, x)
    hv_y = compute_gradient(hv, y)
    A = hu_x + hv_y
    B = u_x + v_y
    A_t = compute_gradient(A, t)
    A_x = compute_gradient(A, x)
    A_y = compute_gradient(A, y)
    B_t = compute_gradient(B, t)
    B_x = compute_gradient(B, x)
    B_y = compute_gradient(B, y)

    z_alpha = -0.53 * h_pred + 0.47 * z_pred
    z_alpha_x = compute_gradient(z_alpha, x)
    z_alpha_y = compute_gradient(z_alpha, y)

    # calculate u and v at the water surface elevation
    temp1 = (z_alpha**2/2-1/6*(h_pred**2-h_pred*z_pred+z_pred**2))
    temp2 = (z_alpha+1/2*(h_pred-z_pred))
    u_2 = temp1*B_x + temp2*A_x
    v_2 = temp1*B_y + temp2*A_y
    u_surface = u_pred + u_2
    v_surface = v_pred + v_2
    H = h_pred + z_pred
    Hu_surface = H*u_surface
    Hv_surface = H*v_surface
    Hu_x = compute_gradient(Hu_surface, x)
    Hv_y = compute_gradient(Hv_surface, y)

    # V1, dispersive Boussinesq terms
    V1Ax = z_alpha**2/2*B_x + z_alpha*A_x
    V1Ax_t = compute_gradient(V1Ax, t)
    V1Ay = z_alpha**2/2*B_y + z_alpha*A_y
    V1Ay_t = compute_gradient(V1Ay, t)
    V1B = z_pred**2/2*B_t+z_pred*A_t
    V1Bx = compute_gradient(V1B, x)
    V1By = compute_gradient(V1B, y)
    V1x = V1Ax_t - V1Bx
    V1y = V1Ay_t - V1By
    # V2, dispersive Boussinesq terms
    V2 = (z_alpha-z_pred)*(u_pred*A_x+v_pred*A_y) \
        +1/2*(z_alpha**2-z_pred**2)*(u_pred*B_x+v_pred*B_y)+1/2*(A+z_pred*B)**2
    V2x = compute_gradient(V2, x)
    V2y = compute_gradient(V2, y)
    # V3
    omega0 = v_x - u_y
    omega2 = z_alpha_x * (A_y + z_alpha * B_y) - z_alpha_y * (A_x + z_alpha * B_x)
    V3x = -omega0*v_2 - omega2*v_pred
    V3y = omega0*u_2 + omega2*u_pred

    # loss with physics (Navier Stokes / Boussinesq etc)
    f_cont = z_t + Hu_x + Hv_y # continuity eq.
    f_momx = u_t + u_pred * u_x + v_pred * u_y + 9.81 * z_x + V1x + V2x + V3x   # momentum in X dir
    f_momy = v_t + u_pred * v_x + v_pred * v_y + 9.81 * z_y + V1y + V2y + V3y   # momentum in Y dir
    
    loss_f = torch.mean(f_cont**2) + torch.mean(f_momx**2) + torch.mean(f_momy**2)

    return loss_f