import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial import distance


def generate_histo(data_points):
    plt.hist(data_points, range=[0.0,2.0])
    plt.show()


def run():
    for i in [5, 10, 20, 100, 1000, 100000]:
        l = []
        for j in range(200):
            v = np.zeros(i)
            while norm(v) < .0001:
                v = np.random.normal(0, 1, i)
            v = v / norm(v)
            l.append(v)
        dist = []
        for j in range(len(l)):
            for k in range(j + 1, len(l)):
                dist.append(distance.euclidean(l[j], l[k]))
        generate_histo(dist)


run()
