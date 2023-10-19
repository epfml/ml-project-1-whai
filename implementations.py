import numpy as np
from helpers import batch_iter


def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

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
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

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
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
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

    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss"""

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    assert w.shape[1]==1, "weights not the right shape"

    loss = 0
    for i in range(y.shape[0]):
        loss += -(1 / y.shape[0]) * (
            y[i] * np.log(sigmoid(tx[i, :] @ w))
            + (1 - y[i]) * np.log(1 - sigmoid(tx[i, :] @ w))
        )

    return loss[0]


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)"""

    gradient = 0
    N = y.shape[0]

    for i in range(N):
        gradient += -(1 / N) * (
            y[i]
            * (1 / (sigmoid(tx[i, :] @ w)))
            * sigmoid(tx[i, :] @ w)
            * (1 - sigmoid(tx[i, :] @ w))
            * tx[i, :].T
            - (1 - y[i])
            * (1 / (1 - sigmoid(tx[i, :] @ w)))
            * sigmoid(tx[i, :] @ w)
            * (1 - sigmoid(tx[i, :] @ w))
            * tx[i, :].T
        )

    return gradient.reshape((-1, 1))


def logistic_learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)"""

    gradient = calculate_logistic_gradient(y, tx, w)
    new_w = w - gamma * gradient
    new_loss = calculate_logistic_loss(y, tx, new_w)

    return new_w, new_loss


def logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma):
    # start the logistic regression
    w = initial_w
    loss = calculate_logistic_loss(y, tx, initial_w)

    for n in range(max_iters):
        # get loss and update w.
        w, loss = logistic_learning_by_gradient_descent(y, tx, w, gamma)

    return w, loss


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)"""
    N = y.shape[0]
    S = np.zeros((N, N))
    for n in range(N):
        S[n, n] = sigmoid(tx[n, :].T @ w) * (1 - sigmoid(tx[n, :].T @ w))

    return (1 / N) * tx.T @ S @ tx


def logistic_regression_loss_gradient_hessian(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)"""

    loss = calculate_logistic_loss(y, tx, w)
    gradient = calculate_logistic_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)

    return (loss, gradient, hessian)


def logistic_learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of Newton's method.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)"""
    loss, gradient, hessian = logistic_regression_loss_gradient_hessian(y, tx, w)

    new_w = w - gamma * np.linalg.solve(hessian, gradient)
    new_loss = calculate_logistic_loss(y, tx, new_w)
    return new_w, new_loss


def logistic_regression_newton_method(y, tx, initial_w, max_iters, gamma):
    """
    Compute one step of the the regularized logistic regression, using the newton method.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        max_iter: int
        initial_w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    w = initial_w
    loss = calculate_logistic_loss(y, tx, w)

    for n in range(max_iters):
        w, loss = logistic_learning_by_newton_method(y, tx, w, gamma)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, gd=True):
    """
    Compute the standard logistic regression, with either the gradient descent or the Newton method.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        max_iter: int
        initial_w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar
        gd: bool

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    if gd:
        return logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma)
    return logistic_regression_newton_method(y, tx, initial_w, max_iters, gamma)


"""
REGULARIZED LOGISTIC REGRESSION
"""


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)"""

    logistic_gradient = calculate_logistic_gradient(y, tx, w)
    gradient = logistic_gradient + 2 * lambda_ * w

    loss = calculate_logistic_loss(y, tx, w) + lambda_ * w.T @ w

    return loss[0, 0], gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)"""
    loss, penalized_gradient = penalized_logistic_regression(y, tx, w, lambda_)

    new_w = w - gamma * penalized_gradient
    new_loss = calculate_logistic_loss(y, tx, new_w)
    return new_w, new_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Compute the regularized logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        max_iter: int
        initial_w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    w = initial_w
    loss = calculate_logistic_loss(y, tx, w)

    for n in range(max_iters):
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

    return w, loss


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    n=x.shape[0]
    indices = np.random.permutation(n)
    index_split = int(np.floor(ratio * n))
    index_tr = indices[:index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te