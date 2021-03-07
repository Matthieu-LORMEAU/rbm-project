import numpy as np
from scipy.special import expit
import math
from RBM import RBM


class DNN:
    """Create a deep neural network allowing for pretraining using contrastive divergence. 
    Inputs are flattened to one dimensional arrays.
    """

    def __init__(self, input_shape: tuple[int], layers_sizes: Iterable[int]):
        """Instantiate a DNN

        Parameters
        ----------
        input_shape : tuple[int]
        layers_sizes : Iterable[int]
            List of layers where each value is an integer corresponding to the number of units of each layer
        """
        self.layers = [RBM(input_shape, layers_sizes[0])]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(RBM(layers_sizes[i], layers_sizes[i + 1]))

        self.pretrained = False
        self.trained = False

    def pretrain(self, X, batch_size, num_epochs=100, lr=0.1):
        """Prerain the DNN using contrastive divergence

        Parameters
        ----------
        X : ndarray
            (number of samples, input_size)
        batch_size : int
        num_epochs : int, optional
            number of epochs, by default 100
        lr : float, optional
            learning rate, by default 0.1

        Returns
        -------
        list[float]
            List of reconstruction errors for each epoch ??????????????????????

        Raises
        ------
        ValueError
            Input dimensions must be (n_samples, RBM.input_size)
        """
        if len(X.shape) != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                "Input dimensions must be (n_samples, RBM.input_size)")

        for l in self.layers:
            l.train(X, batch_size, num_epochs, lr)
            X = l.input_output(X)

    def retropropagation(self):
        ...

    def input_output(self):
        """Compute the output for a certain input

        Parameters
        ----------
        input : ndarray
            (number of samples, input_size)

        Returns
        -------
        ndarray:
            (number of samples, last_layer_size)
        """
        return expit(input @ self.W + self.b)

    def generate_image_DBN(self):
        ...

    def test_DNN(self):
        ...
