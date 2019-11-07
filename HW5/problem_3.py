# 3. Eigenfaces (15 pts). In this exercise you will apply PCA to a
# modified version of the Extended Yale Face Database B. The modified
# database is available in the file yalefaces.mat on Canvas. The
# modification was simply to reduce the resolution of each image by
# a factor of 4 4 = 16 to hopefully avoid computational and memory bottlenecks.
# Plot images of a few of the samples. The data consist of several
# different subjects (38 total) under a variety of lighting conditions.

# (a) (5 pts) By viewing each image as a vector in a high dimensional
# space, perform PCA on the full dataset. Do not use a built-in method
# that performs PCA automatically. Hand in a plot of the sorted eigenvalues
# (use the semilogy command in Matlab; plt.semilogy in Python) of
# the sample covariance matrix. How many principal components are needed to represent 95%
# of the total variation? 99%? What is the percentage reduction in dimension in each case?
# Useful commands in Matlab: reshape, eig, svd, mean, diag, and Python: np.reshape,
# np.linalg.eig, np.linalg.svd, np.mean, np.diag.

# (b) (5 pts) Hand in a 4 × 5 array of subplots showing principal eigenvectors
# (‘eigenfaces’) 0 through 19 as images, treating the sample mean as the zeroth order principal eigenvector. Comment
# on what facial or lighting variations some of the different principal components are capturing.
# Useful commands in Matlab: subplot, imagesc, colormap(gray),
# in Python: plt.imshow(x, cmap=plt.get cmap(’gray’)), and for subplots a useful link is
# http://matplotlib.org/examples/pylab examples/subplots demo.html

# (c) (5 pts) Turn in your code.


# Notes:
# 42 subjects
# 48 lighting conditions
# 2414 size images
import numpy as np
from numpy.linalg import eig, svd
from sklearn.decomposition import PCA
from numpy import mean, cov
from scipy.io import loadmat
from matplotlib import pyplot as plt
import pandas as pd


def pca_mine(data, k=-1):
    # center each column with its mean
    # get eign decomposition and covariance matrix
    # get data projections
    mu = mean(data, axis=0)
    dt = data - mu
    cm = cov(dt.T)
    val, vec = np.linalg.eig(cm)
    val = np.real(val / np.sum(val))
    vec = np.real(vec)
    p = np.dot(vec, dt.T)
    pairs = list(zip(val, vec))
    pairs.sort(key=lambda x: x[0], reverse=True)
    va, ve = zip(*pairs)
    sig = p.std(axis=0).mean()
    return p, va, ve, cm, mu, sig


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
faces = loadmat('yalefaces.mat')['yalefaces']
fr = faces.reshape(-1, 2414) / 255.0
p, val, vec, V, mu, sig = pca_mine(fr)
pc = PCA()
ret = pc.fit_transform(fr)

plt.plot(val)
plt.xlabel('Vector')
plt.ylabel('Explained Variance')
plt.show()

fig, ax = plt.subplots()
ax.scatter(vec[0], vec[1])
ax.set_aspect('equal')
plt.show()
plt.semilogy(V)
plt.show()
plt.semilogy(pc.get_covariance())
plt.show()
plt.plot(vec[0])
plt.show()
#
print('hello')

# A: 62 -> 95%, 234 -> 99%
