import numpy as np
from scipy.special import expit
import math
from codes.utils import reconstruction_error, cross_entropy, accuracy_score
from codes.RBM import RBM
from collections import Iterable


class DNN:
    """Create a deep neural network allowing for pretraining using contrastive divergence. 
    Inputs are flattened to one dimensional arrays.
    """
    def __init__(self, input_size: int, layers_sizes: Iterable[int],
                 output_size: int):
        """Instantiate a DNN
        
        Parameters
        ----------
        input_size : int
            number of input features
        layers_sizes : Iterable[int]
            List of layers where each value is an integer corresponding to the number of units of each layer
        output_size : int
            number of output labels
        """
        self.input_size = input_size
        self.layers = [RBM(input_size, layers_sizes[0])]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(RBM(layers_sizes[i], layers_sizes[i + 1]))

        self.classif_RBM = RBM(self.layers[-1].output_size, output_size)
        self.pretrained = False
        self.trained = False

    def softmax(self, rbm: RBM, input):
        """Compute Softmax probabilities.

        Parameters
        ----------
        rbm [RBM]: object
        input [ndarray]: (rbm.input_size,)

        Returns
        -------
        ndarray: 
            (rbm.input_size,)
        """
        output = input @ rbm.W + rbm.b
        return np.exp(output) / np.sum(np.exp(output), axis=0)

    def pretrain(self, X, batch_size, num_epochs=100, lr=0.1):
        """Prerain the DNN using contrastive divergence.

        Parameters
        ----------
        X : ndarray
            (number of samples, input_size)
        batch_size : int
        num_epochs : int, optional
            number of epochs, by default 100
        lr : float, optional
            learning rate, by default 0.1

        Raises
        ------
        ValueError
            Input dimensions must be (n_samples, RBM.input_size)
        """
        if len(X.shape) != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                "Input dimensions must be (n_samples, RBM.input_size)")

        init_X = X.copy()

        # pretraining
        for l in self.layers:
            l.train(X, batch_size, num_epochs, lr, False)
            X = l.input_output(X)

        # reconstruction error
        error = 0
        for x in init_X:
            h_L = self.input_output(x)
            x_tild = self.output_input(h_L)
            error += reconstruction_error(x, x_tild)

        to_print = "Pretraining Complete: Reconstruction Error: {:.7f}".format(
            error / init_X.shape[0])
        print(to_print)

        self.pretrained = True

    def input_output(self, input):
        """Compute the output for a certain input.

        Parameters
        ----------
        input : ndarray
            (number of samples, input_size)

        Returns
        -------
        ndarray:
            (number of samples, last_layer_size)
        """
        for l in self.layers:
            p_h = l.input_output(input)
            input = (np.random.random_sample(l.output_size) < p_h) * 1
        return input

    def output_input(self, output):
        """Compute the input for a certain output.

        Parameters
        ----------
        output : ndarray
            (number of samples, output_size of last RBM)

        Returns
        -------
        ndarray:
            (number of samples, input_size)
        """
        for l in reversed(self.layers):
            p_v = l.output_input(output)
            output = (np.random.random_sample(l.input_size) < p_v) * 1
        return output

    def back_propagation(self, X, Y, batch_size, num_epochs=100, lr=0.1):
        """Descent Gradient Algorithm for DNN

        Parameters
        ----------
        X : ndarray
            (n_samples, dim)
        Y : ndarray
            (n_samples, classes)
        batch_size : int
        num_epochs : int, optional
            number of epochs, by default 100
        lr : float, optional
            learning rate, by default 0.1

        Returns
        -------
        tuple
            list of losses, list of scores for each epoch

        Raises
        ------
        ValueError
            Dimensions error
        """        
        if len(X.shape) != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                "Input dimensions must be (n_samples, RBM.input_size)")

        n_samples = X.shape[0]
        total_score = []
        total_loss = []
        for e in range(num_epochs):
            # shuffle data
            indices = np.random.permutation(n_samples)
            X = X[indices, :]
            Y = Y[indices, :]
            epoch_score = []
            epoch_loss = []
            for b in range(int(np.ceil(n_samples / batch_size))):
                # batch borders
                beg = b * batch_size
                end = min(beg + batch_size, n_samples)
                n = end - beg
                batch = X[beg:end, :]
                targets = Y[beg:end, :]
                layer_wise_output = self.input_output_network(batch)
                predictions = layer_wise_output[-1]
                # cross entropy & score computation
                epoch_score.append(accuracy_score(predictions, targets))
                epoch_loss.append(cross_entropy(predictions, targets))
                # gradients computations
                grads_W = []
                grads_b = []
                # note : layer_wise_output have one more element than self.layers because of self.classif_RBM
                # -- last layer
                C = predictions - targets
                grads_W.append((layer_wise_output[-2].T @ C) / n)
                grads_b.append(np.sum(C) / n)
                # -- over layers
                for layer_idx in reversed(range(len(self.layers))):
                    # layers in reverse order
                    if layer_idx == len(self.layers)-1:
                        C = C @ self.classif_RBM.W.T * (layer_wise_output[layer_idx] * (1-layer_wise_output[layer_idx]))
                    else:
                        C = C @ self.layers[layer_idx+1].W.T * (layer_wise_output[layer_idx] * (1-layer_wise_output[layer_idx]))

                    if layer_idx == 0:
                        grads_W.append(batch.T @ C)
                    else:
                        grads_W.append(layer_wise_output[layer_idx-1].T @ C)
                    grads_b.append(np.sum(C))
                # gradients updates
                for layer_idx in range(len(self.layers)):
                    self.layers[layer_idx].W -= grads_W[-layer_idx-1] * lr/n
                    self.layers[layer_idx].b -= grads_b[-layer_idx-1] * lr/n
                self.classif_RBM.W -= grads_W[0] * lr/n
                self.classif_RBM.b -= grads_b[0] * lr/n

            total_score.append(sum(epoch_score) / len(epoch_score))
            total_loss.append(sum(epoch_loss) / len(epoch_loss))

            if (e % 10) == 0:
                print(
                    "epoch: {0:d} (Loss: train {1:.2f}) (Accuracy: train {2:.2f})"
                    .format(e, total_loss[-1], total_score[-1]))
            
        return total_loss, total_score

    def input_output_network(self, X):
        """Returns the outputs on each hidden layer of 
        the network as well as the probabilities on the output units.

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        list of ndarray
            list of size len(self.layers)
            i_th position corresponds to the ouput of the i_th layer
            in the DNN for every sample
        """
        layer_wise_output = [[] for i in range(len(self.layers)+1)]
        # if X contains only one data
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        # for every data in X
        for x in X:
            for l in range(len(self.layers)):
                p_h = self.layers[l].input_output(x)
                layer_wise_output[l].append(p_h)
                x = (np.random.random_sample(self.layers[l].output_size) <
                     p_h) * 1
            # last layer --> softmax
            layer_wise_output[-1].append(self.softmax(self.classif_RBM, x))

        # just reshape
        layer_wise_output = [[
            layer_wise_output[i][j].reshape(1, -1)
            for j in range(len(layer_wise_output[i]))
        ] for i in range(len(layer_wise_output))]
        # concat results for each layers for each sample
        layer_wise_output = [
            np.concatenate(layer_wise_output[i])
            for i in range(len(layer_wise_output))
        ]
        return layer_wise_output

    def predict(self, X_test):
        """Predict the label for X_test

        Parameters
        ----------
        X_test : ndarray
            (n_samples, data_size)

        Returns
        -------
        ndarray
            (n_samples,)
        """        
        return np.argmax(self.input_output_network(X_test)[-1], axis=1)

    def generate_image_DBN(self, num_images, gibbs_num_iter, reshape=None):
        """Return a list of generated images.

        Parameters
        ----------
        num_images : int
            number of images to generate
        gibbs_num_iter : int
            number of iterations in the Gibbs sampling step

        Returns
        -------
        list[ndarray]
            list of images of shape input_size or reshape (if reshape != None)

        Raises
        ------
        ValueError
            reshape size must be compatible with self.input_size
        """
        generated_images = []

        if reshape is not None:
            if math.prod(reshape) != self.input_size:
                raise ValueError(
                    f"Given reshape {reshape} is incompatible with the input size {self.input_size}."
                )

        for i in range(num_images):
            v = (np.random.random_sample(self.input_size) < 0.5) * 1

            for j in range(gibbs_num_iter):
                h_L = self.input_output(v)
                v = self.output_input(h_L)

            generated_images.append(
                v if reshape is None else v.reshape(reshape))

        return generated_images


def test_DNN(self, dnn: DNN, X_test, Y_test):
    if self.trained:
        ...
