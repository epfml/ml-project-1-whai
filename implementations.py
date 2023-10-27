import numpy as np
from matplotlib import pyplot as plt

from helpers import batch_iter


def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D), D is the number of features.
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    return (1 / 2) * np.mean((y - tx @ w) ** 2)


"""
GRADIENT DESCENT
"""


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D), D is the number of features.
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    gradient = (1 / N) * tx.T @ (tx @ w - y)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_mse_loss(y, tx, w)

    return (w, loss)


"""
STOCHASTIC GRADIENT DESCENT
"""


def compute_stoch_gradient(y, tx, w, batch_size=1):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,D), D is the number of features.
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    mini_batch = next(batch_iter(y, tx, batch_size, num_batches=1, shuffle=True))
    y_batch = mini_batch[0]
    tx_batch = mini_batch[1]

    stoch_gradient = compute_gradient(y_batch, tx_batch, w)
    return stoch_gradient


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
    """

    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - gamma * gradient

    loss = compute_mse_loss(y, tx, w)

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

    w_opt = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse_loss(y, tx, w_opt)

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
    w_opt = np.linalg.solve(
        (tx.T @ tx + 2 * N * lambda_ * np.eye(tx.shape[1])), tx.T @ y
    )
    # rmse    = np.sqrt(np.mean((y - tx@w_opt)**2))
    mse = compute_mse_loss(y, tx, w_opt)

    return (w_opt, mse)


"""
LOGISTIC REGRESSION
"""


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array"""

    return 1.0 / (1 + np.exp(-t.astype(float)))


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        w:  shape=(D,)

    Returns:
        a non-negative loss"""

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    return (
        -1.0
        / y.shape[0]
        * sum(
            y[i] * np.log(sigmoid(tx[i].T @ w))
            + (1 - y[i]) * np.log(1 - sigmoid(tx[i].T @ w))
            for i in range(y.shape[0])
        )
    )


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        w:  shape=(D,)

    Returns:
        a vector of shape (D,)"""

    return 1 / y.shape[0] * tx.T @ (sigmoid(tx @ w) - y)


def logistic_learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        w:  shape=(D,)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D,)"""

    gradient = calculate_logistic_gradient(y, tx, w)
    w -= gamma * gradient
    loss = calculate_logistic_loss(y, tx, w)
    return loss, w


def logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma):
    # start the logistic regression
    w = initial_w
    loss = calculate_logistic_loss(y, tx, initial_w)

    losses = [loss]

    for n in range(max_iters):
        # get loss and update w.
        loss, w = logistic_learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

    plt.plot(losses)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Compute the standard logistic regression, with the gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        max_iter: int
        initial_w:  shape=(D,)
        gamma: scalar
        lambda_: scalar

    Returns:
        w: shape=(D,)
        loss: scalar number
    """

    return logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma)


"""
REGULARIZED LOGISTIC REGRESSION
"""


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        w:  shape=(D,)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D,)"""

    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)
    gradient = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w

    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        w:  shape=(D,)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D,)"""
    loss, penalized_gradient = penalized_logistic_regression(y, tx, w, lambda_)

    w -= gamma * penalized_gradient
    new_loss = calculate_logistic_loss(y, tx, w)
    return w, new_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Compute the regularized logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        max_iter: int
        initial_w:  shape=(D,)
        gamma: scalar
        lambda_: scalar

    Returns:
        w: shape=(D,)
        loss: scalar number
    """
    w = initial_w
    loss = calculate_logistic_loss(y, tx, w)

    for n in range(max_iters):
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

    return w, loss


def my_reg_logistic_regression(
    y_tr, x_tr, y_val, x_val, lambda_, initial_w, max_iters, gamma
):
    """
    Compute the regularized logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D), D is the number of features.
        max_iter: int
        initial_w:  shape=(D,)
        gamma: scalar
        lambda_: scalar

    Returns:
        w: shape=(D,)
        loss: scalar number
    """
    w = initial_w
    loss = calculate_logistic_loss(y_tr, x_tr, w)
    gen_loss = calculate_logistic_loss(y_val, x_val, w)

    losses = [loss]
    gen_losses = [gen_loss]

    for n in range(max_iters):
        w, loss = learning_by_penalized_gradient(y_tr, x_tr, w, gamma, lambda_)
        losses.append(np.abs(loss))
        gen_losses.append(np.abs(calculate_logistic_loss(y_val, x_val, w)))

    return w, losses, gen_losses
