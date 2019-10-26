from scipy.io import loadmat
import numpy as np


def read():
    dict = loadmat('nuclear.mat')
    return np.array(dict['x'].T), np.array(dict['y'].T)


def stochastic_descent(x, y, b, w, e):
    l = 0.001
    for i in range(1, e):
        a = 100 / i
        dw, db = sub_gradient(x, y, w, b, l)
        w = w - a * dw
        b = b - a * db
        print(eval_method(x, y, w, b))
    return w, b


def sub_gradient(x, y, w, b, l):
    dw = 0.0
    db = 0.0
    for i in range(len(x)):
        v = np.dot(w.T, x[i]) + b
        if y[i] * v < 1.0:  # miss classify
            dw += -y[i] * x[i]
            db += -1 * y[i]
        # do not change anything if it classifies correctly
    return dw, db


def eval_method(x, y, w, b):
    acc = 0
    for i in range(len(x)):
        if ((np.dot(x[i], w) + b) * y[i]) > 0:
            acc += 1
    return acc / len(x)


x, y = read()
l = 0.001  # lambda
e = 100  # epochs size
w = np.zeros(len(x[0]*y[0]))
b = np.zeros(len(y[0]))
w, b = stochastic_descent(x, y, b, w, e)
acc = eval_method(x, y, w, b)
print(acc)