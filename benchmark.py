from codes.utils import load_mnist
from codes.DNN import DNN
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_range_or_tqdm(l, leave=False):
    return tqdm(l, leave=leave) if len(l) > 1 else l


def gridtest_params(X_train, y_train, X_test, y_test, num_layers=[2], num_neurons=[200], num_data=[60000], pretrain=[True], train=[True]):

    print("\n######################################## GRIDTEST PARAMS ############################################\n")
    print(
        f"## Num layers : {num_layers}\n## Num neurons : {num_neurons}\n## Num data : {num_data}\n## Pretrain : {pretrain}\n## Train : {train}\n")

    one_hot_y_train = np.zeros((y_train.size, y_train.max()+1))
    one_hot_y_train[np.arange(y_train.size), y_train] = 1
    one_hot_y_test = np.zeros((y_test.size, y_test.max()+1))
    one_hot_y_test[np.arange(y_test.size), y_test] = 1

    tq_data = make_range_or_tqdm(num_data, leave=True)
    for d in tq_data:
        if type(tq_data) != list:
            tq_data.set_description(f"Num samples : {d}")

        indices = np.random.permutation(d)
        cur_X_train, cur_y_train = X_train[indices], one_hot_y_train[indices]

        tq_pre = make_range_or_tqdm(pretrain)
        for pre in tq_pre:
            if type(tq_pre) != list:
                tq_pre.set_description(f"Pretrain : {pre}")

            tq_train = make_range_or_tqdm(train)
            for t in tq_train:
                if type(tq_train) != list:
                    tq_train.set_description(f"Train : {t}")

                tq_layers = make_range_or_tqdm(num_layers)
                for l in tq_layers:
                    if type(tq_layers) != list:
                        tq_layers.set_description(f"Num layers : {l}")

                    tq_neurons = make_range_or_tqdm(num_neurons)
                    for n in tq_neurons:
                        if type(tq_neurons) != list:
                            tq_neurons.set_description(
                                f"Num neurons : {n}")

                        dnn = DNN(cur_X_train.shape[1], [
                                  n for i in range(l)], cur_y_train.shape[1])

                        dnn.pretrain(cur_X_train, 128, num_epochs=100,
                                     lr=0.1, verbose=False)

                        train_loss, test_loss, train_score, test_score = dnn.back_propagation(
                            cur_X_train, cur_y_train, X_test, one_hot_y_test, batch_size=128, num_epochs=100, lr=0.1, verbose=False)

                        res = np.array(
                            [train_loss, test_loss, train_score, test_score], dtype=object).T
                        pd.DataFrame(res, columns=['train_loss', 'test_loss', 'train_score', 'test_score']).to_csv(
                            f"test_outputs/pretrain[{pre}]_layers[{l}]_neurons[{n}]_data[{d}].csv", index=False)

    print("DONE\n")


data = load_mnist(test_data=True)

X_train = data[0][0]
y_train = data[1][0]
X_test = data[0][1]
y_test = data[1][1]

X_train = [X_train[i].flatten().reshape(1, -1)
           for i in range(X_train.shape[0])]
X_train = np.concatenate(X_train)

X_test = [X_test[i].flatten().reshape(1, -1) for i in range(X_test.shape[0])]
X_test = np.concatenate(X_test)

test_layers = [2, 3, 5, 7]
test_neurons = [100, 300, 500, 700]
test_num_data = [3000, 7000, 10000, 30000, 60000]

gridtest_params(X_train, y_train, X_test, y_test,
                num_layers=test_layers, pretrain=[True])
gridtest_params(X_train, y_train, X_test, y_test,
                num_neurons=test_neurons, pretrain=[True])
gridtest_params(X_train, y_train, X_test, y_test,
                num_data=test_num_data, pretrain=[True])
