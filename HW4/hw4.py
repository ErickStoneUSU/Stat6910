import random

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def read():
    dict = loadmat('nuclear.mat')
    d = list(zip(np.array(dict['x'].T), np.array(dict['y'].T)))
    random.shuffle(d)
    return np.array(d).T


def non_stoch_sub_gradient(x, y, w, b, l):
    scores = []
    J = []

    for i in range(len(x)):
        a = 100
        v = np.dot(w.T, x[i]) + b
        if y[i] * v < 1.0:  # miss classify
            w = w - a * y[i] * x[i]
            b = b - a * 1 * y[i]
        # do not change anything if it classifies correctly

        sc = eval_method(x, y, w, b)
        if len(scores) > 0 and scores[-1][1] > 0.93 and abs(sc - scores[-1][1]) < 0.001:
            break

        # print(sc)
        scores.append([i, sc])
        J.append([w, b])
    print(scores)
    return w, b, J, scores


def sub_gradient(x, y, w, b, count, J):
    dw = 0.0
    db = 0.0
    for i in range(len(x)):
        v = np.dot(w.T, x[i]) + b
        if y[i] * v < 1.0:  # miss classify
            jw = -y[i] * x[i]
            jb = -1 * y[i]
            dw += jw
            db += jb
            J.append([w - dw, b - db, count * len(x) + i])

        # do not change anything if it classifies correctly
    return dw, db, J


def stochastic_descent(x, y, b, w, e):
    l = 0.001
    scores = []
    J = []

    for i in range(1, e):
        a = 100 / i
        dw, db, J = sub_gradient(x, y, w, b, i - 1, J)
        w = w - a * dw
        b = b - a * db
        sc = eval_method(x, y, w, b)
        if len(scores) > 0 and scores[-1][1] > 0.93 and abs(sc - scores[-1][1]) < 0.001:
            break

        print(sc)
        scores.append([i, sc])
    return w, b, scores, J


def eval_method(x, y, w, b):
    acc = 0
    for i in range(len(x)):
        if ((np.dot(x[i], w) + b) * y[i]) > 0:
            acc += 1
    return acc / len(x)


def formula(w, b, rmin, rmax):
    # slope intercept form with 2 weights
    # y = (-(b/w1) / (b / w0)) x + (-b / w1)
    res = []
    m = (-(b/w[1]) / (b/w[0]))
    t = (-b/w[1])
    for i in range(rmin, rmax):
        res.append(m * i + t)
    return range(rmin, rmax), res


def plot_data_line(x, y, w, b):
    x0 = [i[0] for i in x]
    x1 = [i[1] for i in x]
    col = ['red' if i == -1 else 'green' for i in y]
    plt.scatter(x0, x1, c=col)
    xl, yl = formula(w, b, 0, 15)
    plt.plot(xl, yl)
    plt.show()


def plot_J(J):
    plt.plot(J[0], J[1])


def main():
    x, y = read()
    l = 0.001  # lambda
    e = 100  # epochs size
    w = np.zeros(len(x[0]))
    b = np.zeros(len(y[0]))
    #w1, b1, scores1, J1 = non_stoch_sub_gradient(x, y, w, b, l)
    w2, b2, scores2, J2 = stochastic_descent(x, y, b, w, e)
    plot_data_line(x, y, w2, b2)
    #print(eval_method(x, y, w1, b1))
    print(eval_method(x, y, w2, b2))
    # 4.B:
    #   figure 1 (data, learned line)
    #   figure 2 (J, iter#)
    plt.plot()


main()