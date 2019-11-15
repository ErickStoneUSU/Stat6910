from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score


def get_data():
    return 1, 2

def run():
    # Part A
    train_x, train_y = get_data()
    loocv = LeaveOneOut()
    n_split = loocv.get_n_splits(train_x)
    model = LogisticRegression()
    res = cross_val_score(model, train_x, train_y, cv=loocv)