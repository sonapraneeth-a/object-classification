import numpy as np


def zca_whitening(image, verbose=False):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert (X.ndim == 2)
    EPS = 10e-5
    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)
    X_white = np.dot(X, W)
    return X_white


def global_contrast_norm(image, lmbda=10, epsilon=0.001, verbose=False):
    X = np.array(image)
    # replacement for the loop
    X_average = numpy.mean(X)
    print('Mean: ', X_average)
    X = X - X_average
    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmbda + np.mean(X ** 2))
    X = s * X / np.max(contrast, epsilon)
    return X

