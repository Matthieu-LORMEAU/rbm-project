import numpy as np
from scipy.special import expit
import math
from codes.RBM import RBM
from codes.utils import reconstruction_error
from collections import Iterable


class DNN:
    """Create a deep neural network allowing for pretraining using contrastive divergence. 
    Inputs are flattened to one dimensional arrays.
    """

    def __init__(self, input_size: int, layers_sizes: Iterable[int]):
        """Instantiate a DNN

        Parameters
        ----------
        input_size : int
            number of input features
        layers_sizes : Iterable[int]
            List of layers where each value is an integer corresponding to the number of units of each layer
        """
        self.input_size = input_size
        self.layers = [RBM(input_size, layers_sizes[0])]
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

        to_print = "Pretraining Complete: Reconstruction Error: {:.7f}".format(error/init_X.shape[0])
        print(to_print)
        
        self.pretrained = True

    def input_output(self, input):
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
        for l in self.layers:
            p_h = l.input_output(input)
            input = (np.random.random_sample(l.output_size) < p_h) * 1
        return input

    def output_input(self, output):
        """Compute the input for a certain output

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

    def retropropagation(self):
        ...

    def generate_image_DBN(self, num_images, gibbs_num_iter, reshape=None):
        """Return a list of generated images

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
                    f"Given reshape {reshape} is incompatible with the input size {self.input_size}.")

        for i in range(num_images):
            v = (np.random.random_sample(self.input_size) < 0.5) * 1

            for j in range(gibbs_num_iter):
                h_L = self.input_output(v)
                v = self.output_input(h_L)

            generated_images.append(
                v if reshape is None else v.reshape(reshape))

        return generated_images

    def test_DNN(self):
        if self.trained:
            ...
