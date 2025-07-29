import numpy as np
from dl_learn import tools, mnist, autodiff
import matplotlib.pyplot as plt

from collections import OrderedDict

class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = autodiff.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = autodiff.ReluLayer()
        self.layers['Affine2'] = autodiff.Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = autodiff.SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = tools.sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = tools.softmax(a2)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = tools.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = tools.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = tools.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = tools.numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1.0
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())

        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

    # TODO: check diff with gradient
    def gradient1(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = tools.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = tools.softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = tools.sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    lr = 0.1

    print(f"x train shape: {x_train.shape}")
    iter_per_epoch = max(train_size / batch_size, 1)

    net = SimpleNet(784, 50, 10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = net.gradient(x_batch, t_batch)
        grad1 = net.gradient1(x_batch, t_batch)

        for key in grad.keys():
            diff = np.average(np.abs(grad[key] - grad1[key]))
            print(key + ":" + str(diff))

        for key in ['W1', 'b1', 'W2', 'b2']:
            net.params[key] -= lr * grad[key]

        loss = net.loss(x_batch, t_batch)
        print(f"idx: {i} loss: {loss}")
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            print(f"i: {i} epoch check")
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)


    x = list(range(len(train_loss_list)))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(x, train_loss_list)
    ax1.set_xlim(xmin=0)
    ax1.set_ylim(ymin=0)

    x1 = list(range(len(train_acc_list)))
    print(f"x1: {x1} iter_per_epoch: {iter_per_epoch}")
    ax2.plot(x1, train_acc_list)
    ax2.plot(x1, test_acc_list)
    ax2.set_xlim(xmin=0)
    ax2.set_ylim(ymin=0)
    plt.show()
