import random
from collections import deque

from scipy.io import loadmat
import numpy as np
from HW3 import Image


# sigmoid function: a is short for activation
def get_a(x, w, b):
    z = np.dot(w, x) + b
    if z < -3:
        return np.array(0.0)
    return 1.0 / (1.0 + np.exp(-z))


# follow the typical structure of sgd, but change the weight update function
def update(mini_batch, w, b, l):
    chb = 0.0
    chw = np.ones(w.shape) / w.shape[0]
    for x, y in mini_batch:
        db, dw = back_prop(x, y, w, b, l)
        chb = chb - db
        chw = chw - dw
    w = w - (1 / len(mini_batch)) * chw
    b = b - (1 / len(mini_batch)) * chb
    return w, b


def back_prop(x, y, w, b, l):
    a = np.array(get_a(x, w, b))
    g = x * (a - y)  # this is the normal gradient change
    h = x.dot(x.T) * a * (1 - a) * x  # the hessian gradient change
    reg = l * np.linalg.norm(a)**2
    dw = g * h + reg
    db = (a - y) * a * (1 - a)
    return db, dw


def eval(mini_batch, w, b):
    s = 0
    for x, y in mini_batch:
        s += get_a(x, w, b) > 0.0 and y == 1
        s += get_a(x, w, b) < -0.0 and y == -1
    return s / len(mini_batch)


def newton(x, y, l, m, iterations):
    w = np.zeros(len(x[0]))
    for i in range(len(w)):
        w[i] += random.uniform(0,1)

    for k in range(iterations):
        t = list(zip(x, y))
        random.shuffle(t)
        mini_batches = [t[i:i + m] for i in range(0, len(t), m)]
        b = 0.01
        for i in mini_batches:
            w_old = w
            w, b = update(i, w, b, l)
            # early stopping if no training difference
            if np.abs(sum(w_old) - sum(w)) < 0.004:
                return w, b
        print(eval(t, w, b))
    return w, b


def read():
    dict = loadmat('mnist_49_3000.mat')
    x = np.array(dict['x']).T
    y = np.array(dict['y']).T
    d = list(zip(x, y))
    random.shuffle(d)
    d = np.array(d).T
    return d[0].T, d[1]


def get_error(a, y):
    # the difference between the value and the expected value
    return np.abs(a - y) ** 2


def get_most_confident(w, b, x, y):
    # confidence equation
    top_confidence = deque()
    for i, j in zip(x, y):
        a = get_a(i, w, b)
        e = (np.abs(j) - np.abs(a))**2  # distance from ideal as error
        confidence = e - 1.96 * np.sqrt(e * (1 - e) / len(i))  # 1.96 is 95% confidence in z table

        # did we guess wrongly?
        if (a > 0) and (j == -1):
            top_confidence.append([confidence, i, j])
        elif (a < 0) and (j == 1):
            top_confidence.append([confidence, i, j])

    # sort by top confidence
    top_confidence = list(top_confidence)
    top_confidence = sorted(top_confidence, key=lambda x: np.sum(x[1]))
    # return top 5
    return top_confidence[0:20]


# 1. get the data
# 2. split into training and validation
# 3. get the weight and bias using the newton classifier
# 4. evalulate accuracy
# 5. get the top 20 most confident wrong answers
# 6. display
def main():
    x, y = read()

    x_t = x[:2000]
    y_t = y[:2000]

    x_v = x[2001:]
    y_v = y[2001:]

    l = 10.0  # lambda
    m = 20  # minibatch size

    w, b = newton(x_t, y_t, l, m, 100)
    print(str(eval(list(zip(x_v, y_v)), w, b)))
    conf = get_most_confident(w, b, x, y)
    Image.plot_images(conf, 4, 5)

main()