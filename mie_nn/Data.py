import torch
import numpy as np
import matplotlib.pyplot as plt
import pymiediff as pmd
import time

def py_Mie_Diff_Scattering(wavelength, diameter, n, k, environment_value = 1.00):
    """
    Inputs: wavelength, diameter, n, k, environment_value (1.00 by default)
    Outputs: angle_scattering (181 values of light scattering intensity per angle)

    Wrapper for pyMieDiff scattering. Used for generating data for analysis.
    """
    # - setup the particle
    wl0 = torch.tensor([wavelength]) # converting to tensor since pmd requires that
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
    """
    Inputs: n_norm, k_norm, wavelength_norm, diameter_norm, theta_norm
    Outputs: log-scaled angle_scattering

    Another wrapper for pyMieDiff scattering. Used to generate data for training.
    
    """

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
    
    scattering = particle.get_angular_scattering(k0=k0, theta=theta_rad).detach().numpy()
    return torch.log10(scattering['i_unpol'])

def generate_particles(N_PARTICLES=1000, N_ANGLES=181, save=True):
    """
    Inputs: N_PARTICLES, N_ANGLES, save
    Outputs: particle_inputs, particle_outputs, particle_derivatives
    
    Generate Mie Scattering Data + jacobian derivatives for N random Particles
    Stores inputs/derivatives in column order [n, k, wavelength, diameter, theta].
    """

    # Particle-level storage: shape (N_PARTICLES, N_ANGLES, ...)
    particle_inputs = np.zeros((N_PARTICLES, N_ANGLES, 5)) # 5 params
    particle_outputs = np.zeros((N_PARTICLES, N_ANGLES)) # one output
    particle_derivatives = np.zeros((N_PARTICLES, N_ANGLES, 5)) # full jacobian storage (diameter kept as placeholder)

    print(f"Generating {N_PARTICLES} particles with derivatives")
    start_time = time.time()

    for p in range(N_PARTICLES):
        wavelength_norm = torch.tensor(np.random.uniform(0, 1))
        diameter_norm   = torch.tensor(np.random.uniform(0, 1))
        n_norm          = torch.tensor(np.random.uniform(0, 1))
        k_norm          = torch.tensor(np.random.uniform(0, 1))
        theta_norm      = torch.tensor(np.linspace(0, 1, N_ANGLES)) # generates 180 angles linearly spaced

        # Compute jacobian derivatives with autograd
        # jacobian wraps inputs with gradient tracking
        jacobian_derivative = torch.autograd.functional.jacobian(
            compute_log_scattering,
            (n_norm, k_norm, wavelength_norm, diameter_norm, theta_norm),
            vectorize=True,
        )

        # compute actual values without gradients for storage
        with torch.no_grad():
            log_scattering = compute_log_scattering(n_norm, k_norm, wavelength_norm, diameter_norm, theta_norm)
        
        # Store inputs
        particle_inputs[p, :, 0] = n_norm
        particle_inputs[p, :, 1] = k_norm
        particle_inputs[p, :, 2] = wavelength_norm
        particle_inputs[p, :, 3] = diameter_norm
        particle_inputs[p, :, 4] = theta_norm
        
        # Store outputs
        particle_outputs[p, :] = log_scattering.detach().numpy()
        
        # Store derivatives
        particle_derivatives[p, :, 0] = jacobian_derivative[0].detach().numpy()           # delta scattering/delta n
        particle_derivatives[p, :, 1] = jacobian_derivative[1].detach().numpy()           # delta scattering/delta k
        particle_derivatives[p, :, 2] = jacobian_derivative[2].detach().numpy()           # delta scattering/delta wavelength
        particle_derivatives[p, :, 3] = jacobian_derivative[3].detach().numpy()           # delta scattering/delta diameter
        particle_derivatives[p, :, 4] = jacobian_derivative[4].diag().detach().numpy()    # delta scattering/delta theta (diagonal)

        if p % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  particle {p}/{N_PARTICLES}  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - start_time
    print(f"\nDone! Total time: {total_time:.1f}s ({total_time/N_PARTICLES:.2f}s per particle)")

    if save:
        np.save('inputs_5d.npy', particle_inputs.reshape(-1, 5))
        np.save('outputs_5d.npy', particle_outputs.reshape(-1, 1))
        np.save('derivatives_5d.npy', particle_derivatives.reshape(-1, 5))
        print("\nSaved data")

    return particle_inputs, particle_outputs, particle_derivatives

def load_and_prepare_training_data(training_split=0.8, verbose = True):
    """
    Load the saved .npy files. Normalizes inputs to [0, 1], log-transform outputs.
    Returns X_train, y_train, dy)train, X_test, y_test, dy_test (all torch tensors)
    """

    inputs_saved = np.load('inputs_5d.npy')             # (N, 5)
    outputs_saved = np.load('outputs_5d.npy')           # (N, 1)
    derivatives_saved = np.load('derivatives_5d.npy')   # (N, 5)

    N_SAMPLES = inputs_saved.shape[0]
    print(f"Loaded {N_SAMPLES} training samples")

    n_train = int(training_split * N_SAMPLES)

    # splitting data between training and testing sets
    x_train = torch.FloatTensor(inputs_saved[:n_train])
    y_train = torch.FloatTensor(outputs_saved[:n_train])
    dy_train = torch.FloatTensor(derivatives_saved[:n_train])

    x_test = torch.FloatTensor(inputs_saved[n_train:])
    y_test = torch.FloatTensor(outputs_saved[n_train:])
    dy_test = torch.FloatTensor(derivatives_saved[n_train:])

    # describing 
    if verbose:
        print(f"Loaded {N_SAMPLES} samples")
        print(f"  X_train: {x_train.shape}, y_train: {y_train.shape}, dy_train: {dy_train.shape}")
        print(f"  X_test:  {x_test.shape}, y_test:  {y_test.shape}, dy_test:  {dy_test.shape}")
        print(f"  Sample input (normalized): {x_train[0]}")
        print(f"  Sample derivative (truth): {dy_train[0]}")
 
    return x_train, y_train, dy_train, x_test, y_test, dy_test



