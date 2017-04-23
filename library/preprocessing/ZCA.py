import numpy as np


class ZCAWhiten:

    def __init__(self):
        self.verbose = True

    def fit_transform(X):
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

    def compute_zca_transform(imgs, filter_bias=0.1):
        meanX = np.mean(imgs, 0)
        covX = np.cov(imgs.T)
        D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))
        assert not np.isnan(D).any()
        assert not np.isnan(E).any()
        assert D.min() > 0
        D = D ** -0.5
        W = np.dot(E, np.dot(np.diag(D), E.T))
        return meanX, W

    def zca_whiten(train, test, cache=None):
        if cache and os.path.isfile(cache):
            with open(cache, 'rb') as f:
                (meanX, W) = pickle.load(f)
        else:
            meanX, W = compute_zca_transform(train)

            with open(cache, 'wb') as f:
                pickle.dump((meanX, W), f, 2)
        train_w = np.dot(train - meanX, W)
        test_w = np.dot(test - meanX, W)
        return train_w, test_w