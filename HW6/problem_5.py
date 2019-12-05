from scipy.io import loadmat
from sklearn.model_selection import LeaveOneOut, cross_val_score, GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
pi_2_sqrt = np.sqrt(np.pi * 2)


def get_data():
    dict = loadmat('anomaly.mat')
    # get x,y
    x = dict['X']
    x_t1 = dict['xtest1']
    x_t2 = dict['xtest2']
    return x.T, x_t1.T, x_t2.T


def gau(x, bandwidth):
    return np.exp(-x**2/(2*bandwidth**2))/(bandwidth*pi_2_sqrt)


def part_a(x):
    loocv = LeaveOneOut()
    loocv.get_n_splits(x)

    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=loocv)
    grid.fit(x)
    print(grid.best_params_)


def part_b(x):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)
    samps = np.random.uniform(-2, 4, 1000)
    dens = kde.score_samples(samps.reshape(-1,1))
    plt.plot(samps, dens)
    plt.show()
    print('b')


def part_d_test(x, p1, p2):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)
    sc1 = kde.score(p1)
    sc2 = kde.score(p2)
    sc_b = cross_val_score(kde, x)
    sc1_b = sc1 / (sum(sc_b) / len(sc_b))
    sc2_b = sc2 / (sum(sc_b) / len(sc_b))
    print(sc1_b)
    print(sc2_b)
    print('c')


def get_score1(x, p):
    # get the f'h(x) :: KDE
    # k_h: bandwidth^-d k (y/bandwidth)
    k_h = gau(x, .1)
    # KDE: 1/n * sum(k_h(X-x))
    kde = 1/len(x) * np.sum(k_h * (x - p))

    # get the average of all the points except the one you are testing
    av = sum(x) / len(x)
    # divide the f'h(x) by the average to get the outlier score
    return np.abs(kde / av)


def part_d(x, p1, p2):
    print(get_score1(x, p1))
    print(get_score1(x, p2))


def kde(x, y, bandwidth):
    kde_v = gau(x, bandwidth)/len(x)


def get_score2(x, p, k):
    # get k nearest neighbors
    # divide dist to furthest knn by ave dist to knn
    dist = [np.abs(i - p) for i in x]
    dist = sorted(dist)[:k]
    return dist[-1] / (sum(dist) / k)


def part_f(x, p1, p2, k):
    print([k, get_score2(x, p1, k)])
    print([k, get_score2(x, p2, k)])


def run():
    # Part A
    x, x_t1, x_t2 = get_data()
    # part_a(x) # get the bandwidth
    # part_b(x) # plot the kde
    for i in [100, 150, 200]:
        part_f(x, x_t1, x_t2, i)  # The outlier score


run()
