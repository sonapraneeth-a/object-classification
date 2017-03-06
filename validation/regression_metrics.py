import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)


class Regression:

    @staticmethod
    def mse(y_true, y_pred):
        diff = y_true.flatten() - y_pred.flatten()
        error = (1.0 / diff.shape[0]) * np.sum(np.power(diff, 2))
        return error

    @staticmethod
    def rmse(y_true, y_pred):
        diff = y_true.flatten() - y_pred.flatten()
        error = np.sqrt((1.0 / diff.shape[0]) * np.sum(np.power(diff, 2)))
        return error

    @staticmethod
    def r2_score(y_true, y_pred):
        diff_1 = np.power(y_true.flatten() - y_pred.flatten(), 2)
        diff_2 = np.power(y_true.flatten() - np.mean(y_true), 2)
        num = np.sum(diff_1)
        den = np.sum(diff_2)
        error = 1 - (num/den)
        return error

    @staticmethod
    def mean_abs_error(y_true, y_pred):
        diff = np.abs(y_true.flatten() - y_pred.flatten())
        error = (1.0 / diff.shape[0]) * np.sum(diff)
        return error

    @staticmethod
    def med_abs_error(y_true, y_pred):
        diff = np.abs(y_true.flatten() - y_pred.flatten())
        error = np.median(diff)
        return error

    @staticmethod
    def exp_var_score(y_true, y_pred):
        num = np.var(y_true.flatten() - y_pred.flatten())
        den = np.var(y_true.flatten())
        error = 1 - (num/den)
        return error

