from scipy.io import loadmat
import numpy as np


# sigmoid function
def get_a(x, w, b):
    z = w * x + b
    return 1.0 / (1.0 + np.exp(-z))


# the rate of change between the outcome and the expected outcome
# the gradient of the sigmoid function
def get_g(x, y, a):
    # weights and bias
    return np.array([[np.sum((y - a) * x), np.sum((y - a) * 1)]])
    # e = np.exp(-x)
    # return e / (e+1)**2


# the hessian of the sigmoid function
def get_h(x, a):
    return np.array([[np.sum((a * (1 - a)) * x * x), np.sum((a * (1 - a)) * x)],
            [np.sum((a * (1 - a)) * x), np.sum(a * (1 - a))]])
    # e = np.exp(x)
    # return e * (e + 1) / (e+1)**3


def get_w(w, x, y, l, b):
    a = get_a(x, w, b)
    grad = get_g(x, y, a)
    hess = get_h(x, a)
    h = np.linalg.inv(hess)

    # derived formula w = w - G/H + l|@|^2
    ch = np.dot(h, grad.T)
    reg = l * np.dot(a, a.T) / 3000.0
    return w + ch[0][0] + np.min(reg), ch[1][0]


def read():
    dict = loadmat('mnist_49_3000.mat')
    return dict['x'], dict['y']


# G = 1/m sum((h(x)-y)x)
# H = 1/m sum(h(x)(1-h(x))xx^T)
# the G/H cancels out the 1/m
# G/H = sum((h(x)-y)x) / sum(h(x)(1-h(x))xx^T)
def newton(x, y, l, m):
    # weights and biases of neural net
    w = np.zeros([x.shape[0], 1], x.dtype) + 0.001
    b = 0
    threshold = 0.000000001

    while True:
        w_old = w
        w, b = get_w(w, x, y, l, b)
        print(np.sum(np.abs(w_old - w)) / 3000)
        if np.sum(np.abs(w_old - w)) / 3000 < threshold:
            break
    return w


def eval(w, x, y):
    acc = 0
    for i, j in zip(x.T, y.T):
        if (np.dot(i, w) < 0) and (j == -1):
            acc += 1
        elif (np.dot(i, w) > 0) and (j == 1):
            acc += 1
    return acc / 3000


x, y = read()
l = 10.0  # lambda
m = 20  # minibatch size
w = newton(x, y, l, m)
acc = eval(w, x, y)
print(acc)
