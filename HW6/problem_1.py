import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial import distance


def generate_histo(data_points):
    plt.hist(data_points)
    plt.show()


def run():
    for i in [5, 10, 20, 100, 1000]:
        l = []
        for j in range(200):
            v = np.zeros(i)
            while norm(v) < .0001:
                v = np.random.normal(0, 1, i)
            v = v / norm(v)
            print(sum(v*v))
            l.append(v)
        generate_histo(l)

# >>> from scipy.spatial import distance
# >>> distance.euclidean([1, 0, 0], [0, 1, 0])
# 1.4142135623730951
# >>> distance.euclidean([1, 1, 0], [0, 1, 0])
# 1.0
run()
