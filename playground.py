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

if __name__ == "__main__":
    test_gradient_descent_plot()
