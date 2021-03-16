from source.utils import load_mnist
from source.DNN import DNN
import numpy as np
import pandas as pd

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

# Grid Search parameters
layers = [[500, 200], [600, 400, 200], [600, 300, 100]]
epochs_pretrain = [100]
epochs_train = [350]
batch_size = [128, 512]
learning_rate = [0.01, 0.1]

parameters = []
for layer in layers:
    for epoch_pretrain in epochs_pretrain:
        for epoch_train in epochs_train:
            for batch in batch_size:
                for lr in learning_rate:
                    parameters.append(
                        [layer, epoch_pretrain, epoch_train, batch, lr])

print(f"#################### GRID SEARCH ####################")
print('\nTotal combinations :', len(parameters))

for comb_idx in range(len(parameters)):

    # parameters definition
    print('\n'+18*'-'+' Combination n°'+
          str(comb_idx)+' '+18*'-'+"\n")
    layer = parameters[comb_idx][0]
    epoch_pretrain = parameters[comb_idx][1]
    epoch_train = parameters[comb_idx][2]
    batch = parameters[comb_idx][3]
    lr = parameters[comb_idx][4]

    # pretraining
    dnn = DNN(784, layer, 10)
    dnn.pretrain(X_train, batch_size=batch,
                 num_epochs=epoch_pretrain, lr=lr)

    # training
    train_total_loss, train_total_score = dnn.back_propagation(
        X_train, Y_train, batch_size=batch, num_epochs=epoch_train, lr=lr)

    score = dnn.test_DNN(X_test, label_test)
    parameters[comb_idx].append(score)

# results
print(f"#################### GRID SEARCH RESULTS ####################")

results = pd.DataFrame(parameters, columns=['Layer', 'Epochs pretraining',
                                            'Epochs training', 'Batch size', 'Learning Rate', 'Test score (accuracy %)'])
results = results.sort_values(by='Test score (accuracy %)', ascending=False)
print(results)
results.to_csv(f"GridSearch.csv", index=False)
