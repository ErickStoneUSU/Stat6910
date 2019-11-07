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

# Build PCA done
# Plot cm done
# 59 to hit 95% variance
# 201 to hit 99% variance
# 1 - 59/2414 = 97.56% dimension reduction
# 1 - 201/2414 = 91.67% dimension reduction

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
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from numpy import mean, cov
from scipy.io import loadmat
from sklearn.decomposition import PCA


def pca_mine(data, k=-1):
    # center each column with its mean
    # get eign decomposition and covariance matrix
    # get data projections
    mu = mean(data, axis=0)
    dt = normalize(data) - mu
    cm = cov(dt)
    val, vec = np.linalg.eig(cm)
    val = np.real(val / np.sum(val))
    vec = np.real(vec)
    p = np.dot(vec, dt)
    pairs = list(zip(val, vec))
    pairs.sort(key=lambda x: x[0], reverse=True)
    if k > 0:
        va, ve = zip(*(pairs[:k]))
    else:
        va, ve = zip(*pairs)
    sig = p.std(axis=0).mean()
    return p, np.array(va), np.array(ve), cm, mu, sig


def load_data():
    faces = loadmat('yalefaces.mat')['yalefaces']
    return np.rot90(faces.T,k=3, axes=(1,2))


def test():
    fr = load_data()
    p, val, vec, cm, mu, sig = pca_mine(fr.reshape(2414, -1))
    pc = PCA()
    ret = pc.fit_transform(fr.reshape(2414, -1))
    print('')


# A: requires semilogy of covariance matrix
def plot_covmat():
    fr = load_data()
    p, val, vec, cm, mu, sig = pca_mine(fr)
    plt.semilogy(cm)
    plt.xlabel('Vector')
    plt.ylabel('Explained Variance')
    plt.show()


def plot_eign_faces():
    fr = load_data()
    p, val, vec, cm, mu, sig = pca_mine(fr.reshape(2414, -1), 20)
    print(p.shape)
    print(vec.shape)
    print(fr.shape)
    vr = np.array(vec)
    for i in range(20):
        # plot each eigenvector into a subplot
        plt.subplot(4, 5, i + 1)
        plt.imshow(p[i].reshape(48, 42), cmap='Greys')
    plt.show()
    print('')


plot_eign_faces()

