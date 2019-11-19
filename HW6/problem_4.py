from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA


def split_data(x, y):
    x_train = x[:int(len(x) * .8)]
    x_test = x[int(len(x) * .8)+1:]
    y_train = y[:int(len(y) * .8)]
    y_test = y[int(len(y) * .8)+1:]
    return x_train, x_test, y_train, y_test


def get_data(with_pca=False, filename='mnist_49_3000.mat'):
    dict = loadmat(filename)
    x = np.array(dict['x']).T
    y = np.array(dict['y']).T
    if with_pca:
        p = PCA(.95)
        p.fit(x)
        print('PCA feature length: ' + str(len(p.explained_variance_ratio_)))
        x = p.transform(x)
    x_train, x_test, y_train, y_test = split_data(x, y)
    return x_train, x_test, y_train, y_test


def train_mod(mod, x, y, filename, do_save=False):
    score = cross_val_score(mod, x, y, cv=200)
    mod.fit(x, y)
    print('Model Score: ' + str(sum(score) / len(score)))
    x_train, x_test, y_train, y_test = get_data(False, 'mnist_49_1000_shifted.mat')
    mod.predict()

    if do_save:
        # save(mod, filename)
        print('hello')


def train(with_pca=False):
    x_train, x_test, y_train, y_test = get_data(with_pca)
    lda_m = LinearDiscriminantAnalysis()
    logreg_m = LogisticRegression(C=1e5, solver='lbfgs', multi_class='warn')  # 82.125
    svm_m = svm.SVC(C=0.5, kernel='linear')  # 82.83
    knn_m = KNeighborsClassifier()
    forest_m = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, min_samples_leaf=3)  # 84.08

    train_mod(lda_m, x_train, y_train, 'lda.pck')
    train_mod(logreg_m, x_train, y_train, 'logreg.pck')
    train_mod(svm_m, x_train, y_train, 'svm.pck')
    train_mod(knn_m, x_train, y_train, 'knn.pck')
    train_mod(forest_m, x_train, y_train, 'forest.pck')


print("A")
train(False)
print("B")
train(True)
