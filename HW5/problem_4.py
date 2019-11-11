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
#   ::: t = 25, chosen t_lower = 19, chosen t_higher = 31

# (c) (4 pts) Apply k-means clustering to just the features of the training data (do not include labels).
# Compute the adjusted Rand index (ARI) between your cluster outputs and the true labels of the
# data points and report the value. Choose one of the PHATE plots from part (a) (a specific value
# of t) and plot the PHATE visualization colored by the cluster labels. Based on the visualization
# and the ARI value, does k-means match the true labels well?
# Note: You may need to do subsampling to make this computationally feasible. If you do, repeat
# the clustering for multiple (say 10-20) random subsamples and report the average ARI. For the
# PHATE plot, choose one of the subsamples and show the results on that.
#    ::: ARI: .34, Graphically, there are significant portions that do not match up.


# (d) (4 pts) Repeat part (c) for the test data and comment on any similarities and differences between
# the results of the two datasets.
#    ::: ARI: .32, Graphically, there are significant portions that do not match up.

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
#   ::: ARI:

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

import pickle

import numpy as np
import phate
from mnist import MNIST
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score

n_clust = 10


def shuff(x, y):
    return np.array(x), np.array(y)
    # z = list(zip(x, y))
    # random.shuffle(z)
    # x_s, y_s = zip(*z)
    # return np.array(x_s), np.array(y_s)


def get_data():
    # note that the file names need to have the . replaced with -
    data = MNIST('data')
    x_train, y_train = data.load_training()
    x_test, y_test = data.load_testing()

    x_train, y_train = shuff(x_train, y_train)
    x_test, y_test = shuff(x_test, y_test)

    return x_train, y_train, x_test, y_test


def pick(d, file_name):
    with open('pickles/' + file_name + str(n_clust) + '.pck', 'wb') as f:
        pickle.dump(d, f)

def load(file_name):
    with open('pickles/' + file_name + str(n_clust) + '.pck', 'rb') as f:
        return pickle.load(f)


def populate_phate(d, name, t=-1):
    ph = None
    if t < 0:
        ph = phate.PHATE(n_components=n_clust)
    else:
        ph = phate.PHATE(t=t, n_components=n_clust)
    cells = ph.fit_transform(d)
    pick(ph, name)
    return ph, np.array(cells).T


def plot_phate(c, c_l, c_h, type, val, labels):
    phate.plot.scatter2d(c, c=labels, title='Optimal T: ' + str(val) + type, filename='op_' + type)
    phate.plot.scatter2d(c_l, c=labels, title='Low T: ' + str(val - 6) + type, filename='low_' + type)
    phate.plot.scatter2d(c_h, c=labels, title='High T: ' + str(val + 6) + type, filename='high_' + type)


def prob4_pop_picks():
    x_train, y_train, x_test, y_test = get_data()

    ph_op, cells_op = populate_phate(x_train, 'op')
    ph_low, cells_low = populate_phate(x_train, 'low', ph_op.optimal_t - 6)
    ph_high, cells_high = populate_phate(x_train, 'high', ph_op.optimal_t + 6)
    print('Optimal T Train: ' + str(ph_op.optimal_t))
    ph_op_test, cells_op_test = populate_phate(x_test, 'op_test')
    ph_low_test, cells_low_test = populate_phate(x_test, 'low_test', ph_op_test.optimal_t - 6)
    ph_high_test, cells_high_test = populate_phate(x_test, 'high_test', ph_op_test.optimal_t + 6)
    print('Optimal T Train: ' + str(ph_op_test.optimal_t))
    # what are the similarities and differences between them?


def prob4_a_plot():
    x_train, y_train, x_test, y_test = get_data()

    op = load('op')
    low = load('low')
    high = load('high')
    op_t = load('op_test')
    low_t = load('low_test')
    high_t = load('high_test')

    plot_phate(op, low, high, 'Train', op.optimal_t, y_train)
    plot_phate(op_t, low_t, high_t, 'Test', op_t.optimal_t, y_test)


def mod_compare(mod, x, y):
    predicted = mod.fit_predict(x)
    ari = adjusted_rand_score(y, predicted)
    print('KMeans ARI: ' + str(ari))
    return mod, predicted, ari


def prob4_kmeans():
    mod = KMeans(n_clusters=10)
    title = 'KMeans'
    x_train, y_train, x_test, y_test = get_data()
    k, p, a = mod_compare(mod, x_train, y_train)
    k_t, p_t, a_t = mod_compare(mod, x_test, y_test)
    op = load('op')
    op_t = load('op_test')
    phate.plot.scatter2d(op, c=p, title=title + ' T: ' + str(op.optimal_t), filename='KMeans_op')
    phate.plot.scatter2d(op_t, c=p_t, title=title + ' Test T: ' + str(op_t.optimal_t), filename='KMeans_op_t')
    print(a)
    print(a_t)


def spec_mod_compare(mod, x, y):
    z = list(zip(x, y))
    ari = 0.0
    j = 0
    preds = []

    try:
        for i in range(0, len(z), 1000):
            xs, ys = zip(*z[i:i+1000])
            mod.fit(xs)
            ari += adjusted_rand_score(ys, mod.labels_)
            preds += list(mod.labels_)
            j += 1
    except Exception as e:
        print('Done')
    return mod, preds, ari / j


def spec_clust():
    # rbf is Gaussian which is default
    x_train, y_train, x_test, y_test = get_data()
    mod = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")
    k, p, a = spec_mod_compare(mod, x_train, y_train)
    k_t, p_t, a_t = spec_mod_compare(mod, x_test, y_test)
    op = load('op')
    op_t = load('op_test')
    phate.plot.scatter2d(op, c=p, title='Spectral T: ' + str(op.optimal_t), filename='Spec_op')
    phate.plot.scatter2d(op_t, c=p_t, title='Spectral Test T: ' + str(op_t.optimal_t), filename='Spec_op_t')
    print(a)
    print(a_t)

# prob4_pop_picks()
# prob4_a_plot()
prob4_kmeans()
spec_clust()
