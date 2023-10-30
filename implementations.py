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

    return 1.0 / (1 + np.exp(-t))


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

    n = x.shape[0]
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


def make_predictions_linear_model(x, w, threshold=0.5, apply_sigmoid=False):
    w2 = w.ravel()
    y_pred = x.dot(w2.T)
    if apply_sigmoid:
        y_pred = sigmoid(y_pred)
    y_pred = np.array([0 if prediction < threshold else 1 for prediction in y_pred])
    return y_pred


def make_predictions_logistic_regression(x, w, threshold=0.5):
    y_pred = sigmoid(x @ w)
    y_pred = np.array([0 if prediction < threshold else 1 for prediction in y_pred])
    return y_pred


def compute_scores_linear_model(x, w, y, threshold=None, apply_sigmoid=False):
    y_pred = make_predictions_linear_model(x, w, threshold, apply_sigmoid)
    TP = np.sum(np.logical_and(y_pred == 1, y == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y == 1))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_scores_logistic_regression(x, w, y, threshold=None):
    y_pred = make_predictions_logistic_regression(x, w, threshold)
    TP = np.sum(np.logical_and(y_pred == 1, y == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y == 1))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    feature_matrix = np.zeros((x.shape[0], degree * x.shape[1]))

    for i in range(1, degree + 1):
        feature_matrix[:, (i - 1) * x.shape[1] : i * x.shape[1]] = x**i

    return feature_matrix


def build_poly_expansion_with_interaction_features(
    x, features_names: list, max_degree, interactions=False
):
    """Build interaction features from x"""
    if interactions:
        poly = np.zeros(
            (x.shape[0], max_degree * x.shape[1] + (x.shape[1] * (x.shape[1] - 1)) // 2)
        )
    else:
        poly = np.zeros((x.shape[0], max_degree * x.shape[1]))

    new_features_name = []
    index = 0
    for j in range(x.shape[1]):
        for degree in range(1, max_degree + 1):
            poly[:, index] = x[:, j] ** degree
            new_features_name.append(features_names[j] + "**" + str(degree))
            index += 1

    if interactions:
        for j in range(x.shape[1]):
            for k in range(j + 1, x.shape[1]):
                poly[:, index] = x[:, j] * x[:, k]
                new_features_name.append(features_names[j] + "*" + features_names[k])
                index += 1

    return poly, new_features_name


def build_k_indices(num_row, k_fold, seed):
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, gamma, max_iters, initial_w):
    val_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].flatten()

    y_val = y[val_indices]
    y_tr = y[tr_indices]
    x_val_cv = x[val_indices]
    x_tr_cv = x[tr_indices]

    w, losses, gen_losses = my_reg_logistic_regression(
        y_tr, x_tr_cv, y_val, x_val_cv, lambda_, initial_w, max_iters, gamma
    )
    loss_tr = losses[-1]
    loss_val = gen_losses[-1]
    return loss_tr, loss_val
