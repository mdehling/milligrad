__all__ = ['mean_squared_error', 'binary_cross_entropy']


def mean_squared_error(y_pred, y_true):
    return pow(y_true-y_pred, 2).mean()

def binary_cross_entropy(y_pred, y_true, *, epsilon=1e-9):
    return - (y_true * (y_pred+epsilon).log() + (1-y_true) * (1-y_pred+epsilon).log()).mean()
