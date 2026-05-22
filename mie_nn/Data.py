import torch
import numpy as np
import matplotlib.pyplot as plt
import pymiediff as pmd
import time

def py_Mie_Diff_Scattering(wavelength, diameter, n, k, environment_value = 1.00):
    # - setup the particle
    wl0 = torch.tensor([wavelength]) # converting into what pyMieDiff requires as input
    k0 = 2 * torch.pi / wl0

    p = pmd.Particle(
        r_layers=[diameter/2],
        mat_layers=[n + k*1j],
        mat_env = 1.00
    )

    theta = torch.arange(0, 181) * torch.pi/181
    
    angle_scattering = p.get_angular_scattering(k0= k0, theta = theta)
    return angle_scattering['i_unpol']


# The blueprint function: takes 5 normalized inputs, returns log scattering
def compute_log_scattering(n_norm, k_norm, wavelength_norm, diameter_norm, theta_norm):
    # Convert normalized to physical values
    wavelength = wavelength_norm * 100 + 450
    diameter   = diameter_norm * 300 + 600
    n          = n_norm * 0.36 + 1.5
    k          = k_norm * 0.32 + 0.68
    theta_rad  = theta_norm * torch.pi
    
    # pyMieDiff setup
    particle = pmd.Particle(
        r_layers=[diameter / 2],
        mat_layers=[n + 1j * k],     # tensor-based complex (preserves gradients)
        mat_env=1.00
    )

    k0 = 2 * torch.pi / wavelength
    
    scattering = particle.get_angular_scattering(k0=k0, theta=theta_rad)
    return torch.log10(scattering['i_unpol'])

