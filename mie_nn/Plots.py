import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import PyMieScatt as ps
import pymiediff as pmd

from Data import py_Mie_Diff_Scattering
from Model import predict_scattering

def plot_losses(train_losses, test_losses, x_train, y_train):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss over training vs Loss over testing for Derivative loss NN")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train min: {y_train.min().item():.4f}")
    print(f"y_train max: {y_train.max().item():.4f}")
    print(f"y_train mean: {y_train.mean().item():.4f}")
    print(f"y_train variance: {y_train.var().item():.4f}")
    print()
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final test loss: {test_losses[-1]:.4f}")