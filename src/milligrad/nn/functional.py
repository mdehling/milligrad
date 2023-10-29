__all__ = [
    'mean_squared_error',
    'binary_cross_entropy',
    'softmax',
    'log_softmax',
]


def mean_squared_error(y_pred, y_true):
    return pow(y_true-y_pred, 2).mean()

def binary_cross_entropy(y_pred, y_true, *, epsilon=1e-9):
    return - (y_true * (y_pred+epsilon).log() + (1-y_true) * (1-y_pred+epsilon).log()).mean()

def softmax(y, axis=-1):
    y = y - y.value.max(axis=axis, keepdims=True)
    exp_y = y.exp()
    return exp_y / exp_y.sum(axis=axis, keepdims=True)

def log_softmax(y, axis=-1):
    y = y - y.value.max(axis=axis, keepdims=True)
    exp_y = y.exp()
    return y - exp_y.sum(axis=axis, keepdims=True).log()
