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
from numpy import mean, cov, select, array
from numpy.linalg import eig
from scipy.io import loadmat


def pca_mine(data):
    # flatten first two dimensions
    # get the mean of each column
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    print(data)
    m = mean(data.T, axis=1)
    # center each column
    c = data - m
    # get covariance matrix
    cm = cov(c)
    # get eign decomposition of covariance matrix
    val, vec = eig(cm)
    # get data projections
    p = vec.T.dot(c)
    return p, val, vec


faces = loadmat('yalefaces.mat')['yalefaces']
test = array([[1, 2], [3, 4], [5, 6]])
p, val, vec = pca_mine(test)