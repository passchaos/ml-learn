import numpy as np
from . import tools

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout

class ReluLayer:
    def __init__(self):
        pass

    def forward(self, x):
        v = x.copy()
        for i in 0..len(v):
            if v[i] < 0:
                v[i] = 0

        v

    def backward(self, dout):
        for i in 0..len(dout):
            if dout[i] < 0:
                dout[i] = 0

        dout

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        return np.dot(dout, self.W.T)

def SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = tools.softmax(x)

        return tools.cross_entropy_error(self.y, t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        return (self.y - self.t) / batch_size
