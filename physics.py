import torch
import numpy as np

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
def continuity_only(x, y, h, U, V):

    hU_x, hV_y = compute_gradient(h*U, x), compute_gradient(h*V, y)

    # Continuity equation loss
    fc = hU_x + hV_y
    loss_continuity = torch.mean(fc**2)

    # Additional condition 1: h = 0.7 when x = 25
    idx = torch.where(x < 25.5)
    loss_condition = torch.mean((h[idx] - 0.75)**2)

    # Total loss
    loss = loss_continuity + loss_condition

    return loss


###########################################################
def continuity_ftemp(x, y, h, U, V):
    
    hU_x, hV_y = compute_gradient(h*U, x), compute_gradient(h*V, y)

    # loss with physics
    fc = hU_x + hV_y  # continuity eq
    
    # loss
    loss = torch.mean(fc**2)

    return loss

###########################################################
def Navier_Stokes(t, x, y, h, z, u, v):
    
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
    Cd = 0.002
    Fs_x = 0 #Cd*abs(U)*u
    Fs_y = 0 #Cd*abs(V)*v
    # breaking forces
    g = 9.81
    gamma_b = 0.78
    Fbr_x = 3.0/16.0 * g * gamma_b**2 * h_x * (h + z)
    Fbr_y = 3.0/16.0 * g * gamma_b**2 * h_y * (h + z)

    # loss with physics (Navier Stokes / Boussinesq etc)
    fc = z_t + hu_x + hv_y                                   # continuity eq
    fm_x = u_t + u * u_x + v * u_y + g * z_x + Fs_x + Fbr_x  # momentum eq in X dir
    fm_y = v_t + u * v_x + v * v_y + g * z_y + Fs_y + Fbr_y  # momentum eq in Y dir

    # loss
    loss = torch.mean(fc**2) + torch.mean(fm_x**2) + torch.mean(fm_y**2)

    return loss

###########################################################
def physics_equation(x, y, h, U, V, eta_mean, Hrms, k):
    
    u_x, u_y = compute_gradient(U, x), compute_gradient(U, y)
    v_x, v_y = compute_gradient(V, x), compute_gradient(V, y)
    z_x, z_y = compute_gradient(eta_mean, x), compute_gradient(eta_mean, y)

    g = 9.81 # m/s2
    rho = 1025 # kg/m3  
    Cd = 0.002 # drag coefficient -> should be higher above the concrete bar

    # Bottom friction
    tau_bx = rho*Cd*U*abs(U)
    tau_by = rho*Cd*V*abs(V)

    # Radiation stresses
    E = 1/8**rho*g*Hrms**2
    Sxx = E * (2*k*h/torch.sinh(2*k*h) + 0.5)
    Syy = E * (1*k*h/torch.sinh(2*k*h) + 0.0)
    Sxx_x, Syy_y = compute_gradient(Sxx, x), compute_gradient(Syy, y)
    Sxy_x, Sxy_y = 0, 0

    # loss with physics
    fc = u_x + v_y                                                                  # continuity eq
    fx = U*u_x + V*u_y + g*z_x + 1/(rho*(eta_mean+h))*(Sxx_x+Sxy_y) + 1/(rho*(eta_mean+h))*tau_bx # momentum eq in X dir
    fy = U*v_x + V*v_y + g*z_y + 1/(rho*(eta_mean+h))*(Sxy_x+Syy_y) + 1/(rho*(eta_mean+h))*tau_by # momentum eq in Y dir

    # loss
    loss = torch.mean(fc**2) + torch.mean(fx**2) + torch.mean(fy**2)

    return loss

