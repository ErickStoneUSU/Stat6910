import numpy as np


def load_data(f):
    z = np.genfromtxt(f, dtype=float, delimiter=',')
    np.random.seed(0)  # Seed the random number generator
    rp = np.random.permutation(z.shape[0])  # random permutation of the indices
    z = z[rp, :]  # shuffle the rows of the data matrix
    x = z[:, :-1]  # 4601 x 57
    y = z[:, -1]

    train_data = x[:2000]
    test_data = x[2000:]

    return train_data, test_data, y


train_data, test_data, y = load_data('spambase.data')
