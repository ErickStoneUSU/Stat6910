import KNN as KNN
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors


def get_data():
    return 1, 2

def run():
    # Part A
    train_x, train_y = get_data()
    lda_m = LinearDiscriminantAnalysis(train_x)
    logreg_m = LogisticRegression(train_x)
    svm_m = svm.SVC(train_x,)
    knn_m = NearestNeighbors(train_x)
    forest_m = RandomForestClassifier(train_x)

    cross_val_score(svm_m, train_x, train_y)  # tuning parameters
    cross_val_score(knn_m, train_x, train_y)  # choose k
    cross_val_score(forest_m, train_x, train_y)  # num of random forests





run()