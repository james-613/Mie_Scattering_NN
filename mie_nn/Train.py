import torch
import torch.nn as nn
import time

def train_model(model, x_train, y_train, dy_train, x_test, y_test, dy_test,
                N_EPOCHS = 200, BATCH_SIZE = 256, lr = 0.0001,
                w_angle = 0.001, w_diameter = 0.1, w_other = 1.0):

    """
    Train with intensity loss weighted derivative losses.

    Derivative loss is split into three groups, each with its own weight:
        - derivative wrt angle (column 4)               weight = 0.001
        - derivative wrt diameter (column 3)            weight = 0.1
        - other (column 0, 1, 2 = n, k, wavelength)     weight = 1

    Total loss = intensity_loss + weight * derivative_wrt_angle_loss + weight * other_loss
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    # for graphing purposes
    train_losses = []
    test_losses = []

    # batching
    n_train = x_train.shape[0]
    n_batches = n_train // BATCH_SIZE

    print(f"Training: {N_EPOCHS} epochs, {n_batches} batches per epoch\n")
    training_start = time.time()
    for epoch in range(N_EPOCHS):
        # Shuffle training data each epoch
        perm = torch.randperm(n_train)# generates a random permutation of the numbers up to 800
        X_shuffled = x_train[perm] # shuffle the training data according to the permutation
        y_shuffled = y_train[perm] # shuffle the 'answers' for the training data
        dy_shuffled = dy_train[perm] # shuffle the derivatives the same way

    # Train one epoch (one pass through the data, in mini-batches)
    epoch_loss = 0.0

    for i in range(n_batches):
        start = i * BATCH_SIZE # index work to see where batch starts
        end = start + BATCH_SIZE # index work to see where each batch ends while looping through the batches
        X_batch = X_shuffled[start:end].clone().requires_grad_(True) # diameters and wavelengths used in this batch
        y_batch = y_shuffled[start:end] # light scattered at each angle in this batch
        dy_batch = dy_shuffled[start:end]

        # forward pass
        pred = model(X_batch)

        # take jacobian derivative of prediction wrt all 5 inputs
        nn_derivative = torch.autograd.grad(pred.sum(), (X_batch), create_graph=True)[0]

        # loss terms
        loss_intensity = loss_fn(pred, y_batch) # get the loss for brightness

        # Compute SEPARATE loss terms for angle vs other derivatives
        loss_derivative_angle = loss_fn(
            nn_derivative[:, 4],     # NN's angle derivative
            dy_batch[:, 4]            # truth angle derivative
        )

        loss_derivative_diameter = loss_fn(
            nn_derivative[:, 3],
            dy_batch[:, 3]
        )
    
        loss_derivative_other = loss_fn(
            nn_derivative[:, [0, 1, 2]],   # NN's n, k, wavelength derivatives
            dy_batch[:, [0, 1, 2]]          # truth n, k, wavelength derivatives
        )

        # Combine with different weights
        loss = loss_intensity + w_angle * loss_derivative_angle + w_diameter * loss_derivative_diameter + loss_derivative_other 
        
        # total loss is intensity loss + derivative loss (scaled down)

        # back propogate
        optimizer.zero_grad() # clears gradients from previous batches
        loss.backward() # back propagation to compute gradients for all weights
        optimizer.step() #updates all weights using the gradients
        
        epoch_loss += loss.item() # records the loss over the batches for graphing
    

    avg_train_loss = epoch_loss / n_batches # average across all 25 batches
    train_losses.append(avg_train_loss) # save avg loss for plotting
    
    # Evaluate on test set (no gradients needed)
    x_test_req = x_test.detach().clone().requires_grad_(True) # this way the original X_test doesn't have to carry around it's claculation history
    test_pred = model(x_test_req) #predict on 200 test examples
    test_nn_derivative = torch.autograd.grad(test_pred.sum(), x_test_req, create_graph=False)[0] # derivative with respect to training parameters
    # print(f"dy_test shape: {dy_test.shape}")
    test_loss_intensity = loss_fn(test_pred, y_test).item() #compute MSE on test set

    # Compute SEPARATE loss terms for angle vs other derivatives
    test_loss_derivative_angle = loss_fn(
        test_nn_derivative[:, 4],     # NN's angle derivative
        dy_test[:, 4]            # truth angle derivative
    )

    test_loss_derivative_diameter = loss_fn(
        test_nn_derivative[:, 3],
        dy_test[:, 3]
    )

    test_loss_derivative_other = loss_fn(
        test_nn_derivative[:, [0, 1, 2]],   # NN's n, k, wavelength derivatives
        dy_test[:, [0, 1, 2]]          # truth n, k, wavelength derivatives
    )

    test_loss = test_loss_intensity + w_angle * test_loss_derivative_angle + w_diameter * test_loss_derivative_diameter + w_other * test_loss_derivative_other
    test_losses.append(test_loss.detach()) # save it
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}  Train Loss: {avg_train_loss:.10f}  Test Loss: {test_loss:.10f}")

    elapsed = time.time() - training_start
    print(f"Total time: {elapsed:.2f}s")
    print(f"Final loss: {train_losses[-1]:.6e}")
    print("\nTraining complete!")

    return train_losses, test_losses
