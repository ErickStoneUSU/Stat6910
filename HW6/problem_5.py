from scipy.io import loadmat
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KernelDensity
import numpy as np
pi_2_sqrt = np.sqrt(np.pi * 2)


def get_data():
    dict = loadmat('anomaly.mat')
    # get x,y
    x = dict['x']
    y = dict['y']
    return x, y


def gau(x, bandwidth):
    return np.exp(-x**2/(2*bandwidth**2))/(bandwidth*pi_2_sqrt)


def test_kde(x, y):
    loocv = LeaveOneOut()
    loocv.get_n_splits(x)

    bandwidth = 1.0
    model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    res = cross_val_score(model, x, y, cv=loocv)
    print(bandwidth)


def kde(x, y, bandwidth):
    kde_v = gau(x, bandwidth)/len(x)


def run():
    # Part A
    train_x, train_y = get_data()
    test_kde(train_x, train_y)


    # Part B

    # Part C
    # The outlier score
    #
