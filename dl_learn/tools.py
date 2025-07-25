import numpy as np

def sigmoid(x: np.ndarray):
    return 1 / ( 1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def step(x: np.ndarray):
    return np.array(x > 0, dtype=np.int8)

def relu(x: np.ndarray):
    return np.maximum(x, 0)

def softmax(x: np.ndarray):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

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

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index

        tmp_value = x[idx]

        x[idx] = tmp_value + h
        v1 = f(x)

        x[idx] = tmp_value - h
        v2 = f(x)

        grad[idx] = (v1 - v2) / (2 * h)
        x[idx] = tmp_value

        it.iternext()

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
