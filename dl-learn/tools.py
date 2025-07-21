import numpy as np

def sigmoid(x: np.ndarray):
    return 1 / ( 1 + np.exp(-x))

def step(x: np.ndarray):
    return np.array(x > 0, dtype=np.int8)

def relu(x: np.ndarray):
    return np.maximum(x, 0)

def softmax(x: np.ndarray):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a
    return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
