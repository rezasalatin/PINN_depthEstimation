# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:32:12 2023

@author: salat
"""

import torch

# Step 1: Set up the tensors
t = torch.rand(64, requires_grad=True)  # A tensor of 64 random elements, requiring gradients
u = torch.rand(64, requires_grad=True)  # A tensor of 64 random elements, requiring gradients

# Step 3: Compute the gradient
u_t = torch.autograd.grad(outputs=u, inputs=t, grad_outputs=torch.ones_like(u), create_graph=True)[0]