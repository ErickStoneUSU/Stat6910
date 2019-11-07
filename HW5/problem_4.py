#####
# Download all of the data for the MNIST dataset from yann.lecun.com/exdb/mnist/. This should
# give you 60,000 images in the training data and 10,000 images in the test data. You will apply PHATE
# to visualize the data and a couple of clustering algorithms to this dataset. You may use any existing
# packages or libraries for this problem as long as you cite them. For all parts, you may assume the same
# number of clusters as classes (10 in this case).
####

####
# (a) (4 pts) Run PHATE on just the features of the training data (do not include labels) using the
# default parameters to obtain a 2D representation of the data. Report the value of t selected using
# the von Neumann entropy (VNE). Rerun PHATE using two different values of t, one value larger
# than the value chosen using the VNE and one value smaller. Plot the PHATE visualization for
# all three values of t (you should end up with 3 different plots) with the data points colored by the
# labels. Comment on the plots. Which of the three values seems to give better separation between
# the classes? Does the relative position of the different classes make sense?
####
#   ::: t = 26, chosen t_lower = 20, chosen t_higher = 32


# (b) (4 pts) Repeat part (a) for the test data and comment on any similarities and differences between
# the results of the two datasets.

# (c) (4 pts) Apply k-means clustering to just the features of the training data (do not include labels).
# Compute the adjusted Rand index (ARI) between your cluster outputs and the true labels of the
# data points and report the value. Choose one of the PHATE plots from part (a) (a specific value
# of t) and plot the PHATE visualization colored by the cluster labels. Based on the visualization
# and the ARI value, does k-means match the true labels well?
# Note: You may need to do subsampling to make this computationally feasible. If you do, repeat
# the clustering for multiple (say 10-20) random subsamples and report the average ARI. For the
# PHATE plot, choose one of the subsamples and show the results on that.

# (d) (4 pts) Repeat part (c) for the test data and comment on any similarities and differences between
# the results of the two datasets.

# (e) (4 pts) Apply spectral clustering to just the features of the training data (do not include labels)
# using a radial or Gaussian kernel. Compute the ARI between your cluster outputs and the true
# labels of the data points. Use the ARI to tune the kernel bandwidth parameter. Report the
# ARI using your selected bandwidth. Choose one of the PHATE plots from part (a) and plot the
# PHATE visualization colored by the cluster labels. Based on the visualization and the ARI value,
# 2
# does spectral clustering do better or worse than k-means?
# Note: You may need to do subsampling to make this computationally feasible. If you do, repeat
# the final clustering for multiple (say 10-20) random subsamples and report the average ARI. For
# the PHATE plot, choose one of the subsamples and show the results on that.

# (f) (4 pts) Apply spectral clustering to the features of the test data using the same kernel and
# bandwidth you selected in part (e). Report the ARI and plot the PHATE visualization colored
# by the cluster labels. Does spectral clustering do better than k-means here?

# (g) (4 pts) Run PHATE on just the features of the training data using the default parameters to
# obtain a 10-dimensional representation of the data. Report the value of t selected using the
# VNE. Apply k-means to the 10-dimensional representation. This can be viewed as a variation on
# spectral clustering. Compute the ARI between your cluster outputs and the true labels of the
# data points. Choose one of the PHATe plots from part (a) and plot the PHATE visualization
# colored by the cluster labels. Based on the visualization and the ARI value, which of the three
# clustering approaches does the best?
# Note: You may need to do subsampling to make this computationally feasible. If you do, repeat
# the clustering for multiple (say 10-20) random subsamples and report the average ARI. For the
# PHATE plot, choose one of the subsamples and show the results on that.

# (h) (4 pts) Repeat part (g) for the test data and comment on any similarities and differences between
# the results of the two datasets.

# (i) (4 pts) Generally when we’re clustering data, we don’t have access to the true labels which makes
# it difficult to tune parameters like the kernel bandwidth for spectral clustering. What is another
#     way you could tune the bandwidth without using cluster or class labels?

# (j) (4 pts) Turn in your code.

# using pyphate to do phate
from mnist import MNIST
import phate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import pickle


def get_data():
    # note that the file names need to have the . replaced with -
    data = MNIST('data')
    x_train, y_train = data.load_training()
    x_test, y_test = data.load_testing()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def pick(d, file_name):
    pickle.dump(d, open(file_name, 'wb'))


def load(file_name):
    return pickle.load(open(file_name, 'rb'))


def populate_phate(d, name, t=-1):
    ph = None
    if t < 0:
        ph = phate.PHATE()
    else:
        ph = phate.PHATE(t=t)
    cells = ph.fit_transform(d)
    plot_phate(ph, cells)
    pick([ph, cells], name + str(t))
    return ph, cells


def plot_phate(ph, cells):
    plt.plot(cells)
    plt.show()


x_train, y_train, x_test, y_test = get_data()

ph_op, cells_op = populate_phate(x_train, 'op')
ph_low, cells_low = populate_phate(x_train, 'low', 20)
ph_high, cells_high = populate_phate(x_train, 'high', 32)

print('hello')

ph_op_test, cells_op_test = populate_phate(x_train, 'op_test')
ph_low_test, cells_low_test = populate_phate(x_train, 'low_test', 20)
ph_high_test, cells_high_test = populate_phate(x_train, 'high_test', 32)

# what are the similarities and differences between them?

# ari = adjusted_rand_score(labels, predicted)
print('hello')
