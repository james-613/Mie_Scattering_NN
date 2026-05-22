import torch
import torch.nn as nn
import numpy as np

def build_model():
    model = nn.Sequential(
        nn.Linear(5, 128),       # input layer: 4 features to 128 neurons
        nn.GELU(),
        nn.Linear(128, 256),     # hidden layer: 128 to 256
        nn.GELU(),
        nn.Linear(256, 128),    # output layer: 256 to 256 (one per angle)
        nn.GELU(),
        nn.Linear(128,1)
    )
    return model

def predict_scattering(model, wavelength, diameter, n, k, angle=None):
    if angle is None:
        angle = np.linspace(0, 180, 181) * np.pi / 180 # 181 angles from 0 to 180 degrees in radians
    angle = np.array(angle)

    # Normalize inputs
    wavelength_norm = (wavelength - 450) / 100
    diameter_norm = (diameter - 600) / 300 
    n_norm = (n - 1.5) / 0.36
    k_norm = (k - 0.68) / 0.32
    theta_norm = angle / np.pi

    input_tensor = torch.FloatTensor([[n_norm, k_norm, wavelength_norm, diameter_norm, theta_norm[i]] for i in range(len(angle))])

    with torch.no_grad():
        log_pred = model(input_tensor)

    pred = 10 ** log_pred.numpy().flatten() # return an array
    return pred
    