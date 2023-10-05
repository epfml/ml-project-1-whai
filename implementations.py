import numpy as np
from helpers import batch_iter

def compute_loss(y, tx, w, MAE=False):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    if MAE:
        loss = np.mean(np.abs(y - tx@w))
    else:
        loss = (1/2)*np.mean((y - tx@w)**2)
    # ***************************************************
    return loss


"""
GRADIENT DESCENT
"""

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    gradient = (1/N)*tx.T@(tx@w - y)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss     = compute_loss(y, tx, w)
        
        w = w - gamma*gradient
        # store w and loss

    return (w, loss)


"""
STOCHASTIC GRADIENT DESCENT
"""


def compute_stoch_gradient(y, tx, w, batch_size=1):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    mini_batch  = next(batch_iter(y, tx, batch_size, num_batches=1, shuffle=True))
    y_batch     = mini_batch[0]
    tx_batch    = mini_batch[1]
    
    stoch_gradient = compute_gradient(y_batch, tx_batch, w)
    return stoch_gradient


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient(y, tx, w, batch_size)
        loss     = compute_loss(y, tx, w)
        
        w = w - gamma*gradient

    return (w, loss)

"""
LEAST SQUARES
"""


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    w_opt   = np.linalg.solve(tx.T@tx, tx.T@y)
    mse     = 0.5*np.mean((y - tx@w_opt)**2)

    return (w_opt, mse)


"""
RIDGE REGRESSION
"""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features."""

    N = len(y)
    w_opt   = np.linalg.solve((tx.T@tx + 2*N*lambda_*np.eye(tx.shape[1])), tx.T@y)
    #rmse    = np.sqrt(np.mean((y - tx@w_opt)**2))
    mse     = 0.5*np.mean((y - tx@w_opt)**2)

    return (w_opt, mse)

"""
LOGISTIC REGRESSION
"""

#To complete


"""
REGULARIZED LOGISTIC REGRESSION
"""

#To complete