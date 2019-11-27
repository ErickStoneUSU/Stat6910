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


def get_data(with_pca=False, filename='mnist_49_3000.mat', shifted=False):
    dict = loadmat(filename)
    x = np.array(dict['x' if not shifted else 'x_shift']).T
    y = np.array(dict['y' if not shifted else 'y_shift']).T
    if with_pca:
        p = PCA(.95)
        p.fit(x)
        print('PCA feature length: ' + str(len(p.explained_variance_ratio_)))
        x = p.transform(x)
    x_train, x_test, y_train, y_test = split_data(x, y)
    return x_train, x_test, y_train, y_test


def train_mod(mod, x, y, x_test, y_test, do_save=False):
    y = y.reshape(-1,)
    score = cross_val_score(mod, x, y, cv=200)
    mod.fit(x, y)
    return sum(score) / len(score), sum(y_test[:,0] == mod.predict(x_test)[:]) / len(x_test)
    # print('Model Score: ' + str(sum(score) / len(score)))


def train(with_pca=False, filename='mnist_49_3000.mat', shifted=False):
    x_train, x_test, y_train, y_test = get_data(with_pca, filename, shifted)
    lda_m = LinearDiscriminantAnalysis()
    logreg_m = LogisticRegression(C=1e5, solver='lbfgs', multi_class='warn')  # 82.125
    svm_m = svm.SVC(C=0.5, kernel='linear')  # 82.83
    knn_m = KNeighborsClassifier()
    forest_m = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, min_samples_leaf=3)  # 84.08

    lda_sc = train_mod(lda_m, x_train, y_train, x_test, y_test, 'lda.pck')
    log_sc = train_mod(logreg_m, x_train, y_train, x_test, y_test, 'logreg.pck')
    svm_sc = train_mod(svm_m, x_train, y_train, x_test, y_test, 'svm.pck')
    knn_sc = train_mod(knn_m, x_train, y_train, x_test, y_test, 'knn.pck')
    forest_sc = train_mod(forest_m, x_train, y_train, x_test, y_test, 'forest.pck')

    return lda_sc, log_sc, svm_sc, knn_sc, forest_sc


print("A")
a = train(False)
print("B")
b = train(True)
print("C_No_PCA")
c = train(False, 'mnist_49_1000_shifted.mat', True)
print("C_With_PCA")
c_pca = train(True, 'mnist_49_1000_shifted.mat', True)
print(a, b, c, c_pca)
