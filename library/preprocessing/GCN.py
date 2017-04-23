import numpy as np


class GCN:


    def __init__(self):
        self.verbose = True
        self.lmbda = 10
        self.epsilon = 0.001

    def fit(X):
        X = np.array(X)
        # replacement for the loop
        X_average = np.mean(X)
        print('Mean: ', X_average)
        X = X - X_average
        # `su` is here the mean, instead of the sum
        contrast = np.sqrt(self.lmbda + np.mean(X ** 2))
        X = s * X / np.max(contrast, self.epsilon)
        return X

    def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
        X = X - X.mean(axis=1)[:, np.newaxis]
        normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
        normalizers[normalizers < min_divisor] = 1.
        X /= normalizers[:, np.newaxis]
        return X
