import csv
import random

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def read_uci_data():
    data = []
    label = []
    header = []
    # https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    # instances: 1372, categories: 2
    # 0 = {str} 'Variance'
    # 1 = {str} 'Skew'
    # 2 = {str} 'Curtosis'
    # 3 = {str} 'Entropy'
    # 0/1 = {int} 'Authentic/Not Authentic'
    with open('data_banknote_authentication.csv') as f:
        r = csv.reader(f, delimiter=',')
        first = True
        for ro in r:
            if first:
                header = [ro[:-1]]
                first = False
            else:
                d = [float(i) for i in ro]
                data.append(np.array(d)[:-1])
                label.append(np.array(float(ro[-1])))

    combined = list(zip(data, label))
    random.shuffle(combined)
    x, y = zip(*combined)

    x_train = np.array(x[:900])
    x_test = np.array(x[901:1000])
    x_val = np.array(x[1001:])
    y_train = np.array(y[:900])
    y_test = np.array(y[901:1000])
    y_val = np.array(y[1001:])
    return x_train, x_test, x_val, y_train, y_test, y_val, header


def logreg(x, y):
    # logistic regression
    mod = LogisticRegression()
    mod.fit(x_train, y_train)
    # k fold cross validation
    scores_train = cross_val_score(mod, x_train, y_train, cv=2)
    scores_test = cross_val_score(mod, x_test, y_test, cv=2)
    return np.average(scores_train), np.average(scores_test)


def svm_t(x, y, typ):
    # describe tuning parameters
    # gaussian kernel svm
    # linear kernel svm
    mod = svm.SVC(kernel=typ, C=1)
    mod.fit(x_train, y_train)
    scores_train = cross_val_score(mod, x_train, y_train, cv=2)
    scores_test = cross_val_score(mod, x_test, y_test, cv=2)
    return np.average(scores_train), np.average(scores_test)


x_train, x_test, x_val, y_train, y_test, y_val, header = read_uci_data()
train_err, test_err = logreg(x_train, y_train)
g_train_err, g_test_err = svm_t(x_train, y_train, 'rbf')
l_train_err, l_test_err = svm_t(x_train, y_train, 'linear')
print('hello')
# which worked best of the three? logreg, gsvm, lsvm
