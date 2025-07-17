# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.arange(0, 6, 0.1)
y = np.sin(x)
y1 = np.cos(x)


plt.plot(x, y, label = "sin")
plt.plot(x, y1, linestyle = "--", label = "cos")

plt.xlabel("x")
plt.ylabel("y")

plt.title("sin & cos")
plt.legend()
plt.show()

# %%
from matplotlib.image import imread
img = imread('DeepLearningFromScratch/dataset/lena.png')

plt.imshow(img)
plt.show()
