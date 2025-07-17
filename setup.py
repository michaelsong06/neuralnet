import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bars while training
from sklearn.datasets import fetch_openml

mnist = fetch_openml(name="mnist_784")
print(mnist.keys())

data = mnist.data
labels = mnist.target

# select random datapoint key from mnist
n = np.random.choice(np.arange(data.shape[0]+1))
print(n)

# get the test image: the image will be in single vector form of shape (784,)
test_img = data.iloc[n].values
test_label = mnist.target.iloc[n]
print(test_img.shape)

# reshape this 1D vector into a 2D 28x28 matrix
side_length = int(np.sqrt(test_img.shape))
reshaped_test_img = test_img.reshape(side_length, side_length)
print("Image label: " + str(test_label))

# show this matrix as an image on matplotlib
plt.imshow(reshaped_test_img, cmap="Greys")
plt.axis('on')
plt.show()