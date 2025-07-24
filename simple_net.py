import numpy as np
from dl_learn import tools

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = tools.softmax(z)

        loss = tools.cross_entropy_error(y, t)

        return loss

if __name__ == "__main__":
    net = SimpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)

    t = np.array([0, 0, 1])
    loss = net.loss(x, t)
    print(f"net loss: {loss}")
