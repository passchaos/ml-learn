import matplotlib.pyplot as plt
import numpy as np
from dl_learn import tools

def function2(x):
    return np.sum(x**2)

def test_numerical_gradient():
    val = tools.numerical_gradient(function2, np.array([2.0, 2.0]))
    print(f"val: {val}")

def gradient_descent_plot_data(f, input, lr=0.01, step_num=100):
    x_arrs = [input[0]]
    y_arrs = [input[1]]

    tmp = input
    for i in range(step_num):
        grad = tools.numerical_gradient(f, input)
        tmp -= lr * grad

        x_arrs.append(tmp[0])
        y_arrs.append(tmp[1])

    return x_arrs, y_arrs

def test_gradient_descent_plot():
    xs, ys = gradient_descent_plot_data(function2, np.array([2.0, 3.0]),
        lr=0.1,
        step_num=100
    )

    plt.scatter(xs, ys)

    t = 3
    plt.xlim(-t, t)
    plt.ylim(-t, t)
    plt.show()

def activation_hist():
    x = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    activations = {}

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]

        w = np.random.randn(node_num, node_num) * np.sqrt(2) / np.sqrt(node_num)

        z = np.dot(x, w)
        # a = tools.sigmoid(z)
        # a = np.tanh(z)
        a = tools.relu(z)
        activations[i] = a

    for i, a in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i+1) + "-layer")
        print(f"a info: {a.flatten().shape}")
        plt.hist(a.flatten(), 30, range=(0, 1))

    plt.show()

if __name__ == "__main__":
    # test_gradient_descent_plot()
    activation_hist()
