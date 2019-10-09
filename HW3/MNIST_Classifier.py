from scipy.io import loadmat
import numpy as np


def sig(z):
    return 1 / (np.exp(-z))


def dsig(sig):
    return sig * (1 - sig)


def ddsig(sig, dsig):
    return sig * dsig * (1 - 2 * sig)


def logl(w, x):
    return np.dot(w, x)


def read():
    dict = loadmat('mnist_49_3000.mat')
    return dict['x'].T, dict['y']


# w = w - grad/Hess
def newton(x, y, l, m):
    w = np.ones(len(x[0])) / len(x[0])  # distribution equivalent to 1
    for i, j in zip(x, y):
        z = np.dot(w.T, i)
        s = sig(z)
        ds = dsig(s)
        reg = l * np.abs(z*z)
        w = w - ds / ddsig(s, ds) - reg
    return w


def eval(w, x, y):
    acc = 0
    for i, j in zip(x, y):
        if (np.dot(i, w) < 0) and (y == -1):
            acc += 1
        elif (np.dot(i, w) > 0) and (y == 1):
            acc += 1
    return acc


x, y = read()
l = 10  # lambda
m = 20  # minibatch size
w = newton(x, y, l, m)
acc = eval(w,x,y)
print(acc)