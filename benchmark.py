from codes.utils import load_mnist
from codes.DNN import DNN
import numpy as np
import pandas as pd


def test_params(X_train, y_train, X_test, y_test, num_layers=[2], num_neurons=[200], num_data=[60000], pretrain=[True], train=[True]):

    for d in num_data:
        indices = np.random.permutation(d)
        X_train, y_train = X_train[indices], y_train[indices]
        one_hot_y_train = np.zeros((y_train.size, y_train.max()+1))
        one_hot_y_train[np.arange(y_train.size), y_train] = 1
        one_hot_y_test = np.zeros((y_test.size, y_test.max()+1))
        one_hot_y_test[np.arange(y_test.size), y_test] = 1
        for pre in pretrain:
            for t in train:
                for l in num_layers:
                    for n in num_neurons:
                        dnn = DNN(X_train.shape[1], [
                                  n for i in range(l)], one_hot_y_train.shape[1])
                        dnn.pretrain(X_train, 128, num_epochs=500, lr=0.1)
                        train_loss, test_loss, train_score, test_score = dnn.back_propagation(
                            X_train, one_hot_y_train, X_test, one_hot_y_test, batch_size=128, num_epochs=500, lr=0.1)

                        res = np.array([train_loss, test_loss, train_score, test_score], dtype=object).T
                        pd.DataFrame(res, columns =['train_loss', 'test_loss', 'train_score', 'test_score']).to_csv(
                            f"test_outputs/pretrain[{pre}]_layers[{l}]_neurons[{n}]_data[{d}].csv", index=False)


data=load_mnist(test_data = True)

X_train=data[0][0]
y_train=data[1][0]
X_test=data[0][1]
y_test=data[1][1]

X_train=[X_train[i].flatten().reshape(1, -1)
           for i in range(X_train.shape[0])]
X_train = np.concatenate(X_train)

X_test = [X_test[i].flatten().reshape(1, -1) for i in range(X_test.shape[0])]
X_test = np.concatenate(X_test)

test_layers = [2, 3, 5, 7]
test_neurons = [100, 300, 500, 700]
test_num_data = [3000, 7000, 10000, 30000, 60000]

test_params(X_train, y_train, X_test, y_test,
            num_layers=test_layers, pretrain=[False, True])
test_params(X_train, y_train, X_test, y_test,
            num_neurons=test_neurons, pretrain=[False, True])
test_params(X_train, y_train, X_test, y_test,
            num_data=test_num_data, pretrain=[False, True])
