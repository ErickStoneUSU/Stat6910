import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.random.seed(1337)
kwargs = {'linewidth' : 3.5}
font = {'weight' : 'normal', 'size'   : 24}
matplotlib.rc('font', **font)


# LASSO is regularized least squares regression
m = 100


# Least Squares Equation ||Ax-b||^2 / 2m
def least_squares(A, b, x):
    """Least squares objective."""
    return (0.5 / m) * np.linalg.norm(A.dot(x) - b) ** 2


def least_squares_gradient(A, b, x):
    """Gradient of least squares objective at x."""
    return A.T.dot(A.dot(x) - b) / m


def lasso(A, b, x, alpha=0.1):
    return least_squares(A, b, x) + alpha * np.linalg.norm(x, 1)


def ell1_subgradient(x):
    """Subgradient of the ell1-norm at x."""
    g = np.ones(x.shape)
    g[x < 0.] = -1.0
    return g


def lasso_subgradient(A, b, x, alpha=0.1):
    """Subgradient of the lasso objective at x"""
    return least_squares_gradient(A, b, x) + alpha * ell1_subgradient(x)


def least_squares_l2(A, b, x, alpha=0.1):
    return least_squares(A, b, x) + (alpha/2) * x.dot(x)

def least_squares_l2_gradient(A, b, x, alpha=0.1):
    return least_squares_gradient(A, b, x) + alpha * x

def gradient_descent(init, steps, grad, proj=lambda x: x):
    """Projected gradient descent.

    Inputs:
        initial: starting point
        steps: list of scalar step sizes
        grad: function mapping points to gradients
        proj (optional): function mapping points to points

    Returns:
        List of all points computed by projected gradient descent.
    """
    xs = [init]
    for step in steps:
        xs.append(proj(xs[-1] - step * grad(xs[-1])))
    return xs

def error_plot(ys, yscale='log'):
    plt.figure(figsize=(8, 8))
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.yscale(yscale)
    plt.plot(range(len(ys)), ys, **kwargs)
    plt.show()


def read():
    dict = loadmat('bodyfat_data.mat')
    x = np.array(dict['X'])
    y = np.array(dict['y'])
    x_train = x[:150]
    x_validate = x[151:]
    y_train = y[:150]
    y_validate = y[151:]
    return x_train, y_train, x_validate, y_validate

from sklearn.linear_model import Lasso
# x is {ab_size,hip_size}
# y is {body fat %}
xt, yt, xv, yv = read()
m, n = 2, 150
A = np.random.normal(0, 1, (m, n))
x_opt = np.zeros(n)
x_opt[:10] = 1.0
b = A.dot(x_opt)
x0 = np.random.normal(0, 1, n)
xs = gradient_descent(x0, [0.1]*500, lambda x: lasso_subgradient(A, b, x))
error_plot([lasso(A, b, x) for x in xs])

plt.figure(figsize=(14,10))
plt.title('Comparison of initial, optimal, and computed point')
idxs = range(50)
plt.plot(idxs, x0[idxs], '--', color='#aaaaaa', label='initial', **kwargs)
plt.plot(idxs, x_opt[idxs], 'r-', label='optimal', **kwargs)
plt.plot(idxs, xs[-1][idxs], 'g-', label='final', **kwargs)
plt.xlabel('Coordinate')
plt.ylabel('Value')
plt.legend()
plt.show()