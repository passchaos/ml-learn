# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import tools
import importlib

importlib.reload(tools)

thre = 6
x = np.arange(-thre, thre, 0.1)
y = tools.sigmoid(x)
y0 = tools.step(x)
y1 = tools.relu(x)

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(x, y, label = "sigmoid")
ax1.plot(x, y0, label = "step")
ax2.plot(x, y1, linestyle = "--", label = "relu")

fig.suptitle("activation function")
fig.legend()
plt.show()

# %%
from matplotlib.image import imread
img = imread('DeepLearningFromScratch/dataset/lena.png')

plt.imshow(img)
plt.show()

# %%
import sys, os
sys.path.append("DeepLearningFromScratch")

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)

# %%
import tools
import numpy as np

# a = np.array([0.2, 0.0, 0.03, 0.07, 0.01, 0.09, 0.3, 0.2, 0.04, 0.06])
a = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
b = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

err1 = tools.mean_squared_error(a, b)
err2 = tools.cross_entropy_error(a, b)

print(f"err: {err1} {err2}")

# %%
import numpy as np
import matplotlib.pyplot as plt
import tools

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
y2 = tools.numerical_diff(function_1, x)

print(f"{y1} {y2}")

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
