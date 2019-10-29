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

    for i in range(1, len(x)):
        a = max(100 / i, 0.1)
        v = np.dot(w.T, x[i]) + b
        if y[i] * v < 1.0:  # miss classify
            dw = -y[i] * x[i]
            db = -1 * y[i]
            w = w - a * dw
            b = b - a * db
        # do not change anything if it classifies correctly

        sc = eval_method(x, y, w, b)
        print(sc)
        if len(scores) > 0 and scores[-1][1] > 0.93 and abs(sc - scores[-1][1]) < 0.001:
            break
        scores.append([i, sc])
        J.append([w, b])
    print(scores)
    return w, b, scores, J


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
            J.append([w - dw, b - db])

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
    m = 0

    if b != 0:
        m = (-(b/w[1]) / (b/w[0]))
    else:
        m = -w[1] / w[0]

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
    #plt.show()


def plot_J_iters(J):
    for i in range(1, len(J), 100):
        xl, yl = formula(J[i][0], J[i][1], 0, 10)
        plt.plot(xl, yl, color=[i / len(J), 0, 0])
    plt.show()


def main():
    x, y = read()
    l = 0.001  # lambda
    e = 100  # epochs size
    w = np.zeros(len(x[0]))
    b = np.zeros(len(y[0]))
    #w1, b1, scores1, J1 = non_stoch_sub_gradient(x, y, w, b, l)
    w2, b2, scores2, J2 = stochastic_descent(x, y, b, w, e)
    plot_data_line(x, y, w2, b2)
    plot_J_iters(J2)
    print(eval_method(x, y, w2, b2))
    # 4.B:
    #   figure 1 (data, learned line)
    #   figure 2 (J, iter#)
    plt.plot()


main()