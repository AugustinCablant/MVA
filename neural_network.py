### imports ###
import numpy as np 

### Activation functions ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

### Forward propagation ###

def forward_propagation(X, W1, b1, W2, b2):
    # Hidden layer
    Z1 = np.dot(X, W1) + b1 
    A1 = relu(Z1)

    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    return A1, A2    # return the activations

### Loss function ###

def compute_loss(y, y_hat, classification = True):
    n = y.shape[0]
    if classification:
        L =  - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / n
    else:
        L = np.sum((y - y_hat)**2) / n
    return L 

### Backward propagation ###
def backward_propagation(X, y, A1, A2, W1, W2):
    
