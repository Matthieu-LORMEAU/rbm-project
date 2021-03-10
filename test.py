from codes.utils import load_mnist
from codes.DNN import DNN
import matplotlib.pyplot as plt
import numpy as np

# load mnist
mnist = load_mnist(train_data=True, test_data=True)
img_train = mnist[0][0]
label_train = mnist[1][0]
img_test = mnist[0][1]
label_test = mnist[1][1]

# flatten images
X_train = [
    img_train[i].flatten().reshape(1, -1) for i in range(img_train.shape[0])
]
X_train = np.concatenate(X_train)
X_test = [
    img_test[i].flatten().reshape(1, -1) for i in range(img_test.shape[0])
]
X_test = np.concatenate(X_test)

# one hot encoding the labels
Y_train = np.zeros((label_train.size, label_train.max() + 1))
Y_train[np.arange(label_train.size), label_train] = 1
Y_test = np.zeros((label_test.size, label_test.max() + 1))
Y_test[np.arange(label_test.size), label_test] = 1

# pretraining
dnn = DNN(784, [128], 10)
indices = np.random.permutation(6000)
dnn.pretrain(X_train[indices], 128, num_epochs=5, lr=0.1)

# training
train_total_loss, test_total_loss, train_total_score, test_total_score = dnn.back_propagation(
    X_train[indices],
    Y_train[indices],
    X_test[:1000],
    Y_test[:1000],
    batch_size=1000,
    num_epochs=100,
    lr=0.1)
