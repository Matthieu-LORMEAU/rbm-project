# from codes.utils import load_mnist
# import matplotlib.pyplot as plt

# train = load_mnist()
# img = train[0][0]
# label = train[1][0]

# plt.imshow(img[0], cmap='gray')
# plt.show()

from codes.utils import load_mnist
from codes.DNN import DNN
import matplotlib.pyplot as plt
import numpy as np

train = load_mnist()
img = train[0][0]
label = train[1][0]

img = [img[i].flatten().reshape(1,-1) for i in range(img.shape[0])]
img = np.concatenate(img)

dnn = DNN(784, [512, 128], 10)
indices = np.random.permutation(60000)
dnn.pretrain(img[indices], 128, num_epochs=5, lr=0.1)

# one hot encoding the labels
one_hot_label = np.zeros((label.size, label.max()+1))
one_hot_label[np.arange(label.size),label] = 1

# backprop
loss, score = dnn.back_propagation(img[indices], one_hot_label[indices], batch_size=512, num_epochs=500, lr=0.1)
