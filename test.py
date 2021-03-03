from codes.utils import load_mnist
import matplotlib.pyplot as plt

train = load_mnist()
img = train[0][0]
label = train[1][0]

plt.imshow(img[0], cmap='gray')
plt.show()
