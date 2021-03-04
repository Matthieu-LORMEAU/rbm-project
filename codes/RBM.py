import numpy as np
from scipy.special import expit
import math


class RBM:
    """Create an RBM object"""

    def __init__(self, input__image_size, output_size):
        self.W = np.random.normal(0, 0.01, (input_size, output_size))
        self.a = np.zeros(input_size)
        self.b = np.zeros(output_size)
        self.input_image_size = input_image_size
        self.input_size = math.prod(input_image_size)
        self.output_size = output_size

    def input_output(self, input):
        # shape data number * outputsize
        return expit(input @ self.W + self.b)

    def output_input(self, output):
        # shape data number * inputsize
        return expit(output @ self.W.T + self.a)

    def reconstruction_error(self, X):
        """
        Return a the reconstruction error score
        """
        X_tild = self.output_input(self.input_output(X))
        return np.linalg.norm(X - X_tild)

    def train(self, X, batch_size, num_epochs=100, lr=0.1):
        """
        Train the RBM object

        Parameters
        ----------

        Return
        ------
        List of reconstruction errors for each epoch
        """

        if len(X.shape) != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                "Input dimensions must be (n_samples, RBM.input_size)")

        n_samples = X.shape[0]
        errors = []

        for e in range(num_epochs):
            # shuffle data
            X = X[np.random.permutation(n_samples), :]

            for b in range(np.ceiling(n_samples / batch_size)):
                # batch borders
                beg = b * batch_size
                end = np.min(beg + batch_size, n_samples)
                n = end - beg
                print(n)
                print(X[beg:end, :].shape)
                # Gibbs sampling one iteration
                v0 = X[beg:end, :]
                prob_h0_v0 = self.input_output(v0)
                h0 = (np.random.random_sample(
                    (n, self.output_size)) < prob_h0_v0) * 1
                prob_v1_h0 = self.output_input(h0)
                v1 = (np.random.random_sample(
                    (n, self.input_size)) < prob_v1_h0) * 1
                prob_h1_v1 = self.input_output(v1)
                # gradients computation
                grad_a = np.sum(v0 - v1, axis=0)
                grad_b = np.sum(prob_h0_v0 - prob_h1_v1, axis=0)
                grad_W = v0.T @ prob_h0_v0 - v1.T @ prob_h1_v1
                # gradients descent
                self.a = self.a + grad_a * lr/n
                self.b = self.b + grad_b * lr/n
                self.W = self.W + grad_W * lr/n

            # reconstruction error
            error = self.reconstruction_error(X)
            errors.append(error)
            epoch_stats = "Epoch {} Complete: Reconstruction Error: {:.7f}".format(e + 1, error)
            print(epoch_stats)

        return errors

    def generate_image(self, num_images, gibbs_num_iter):
        """
        Return a list of generated images
        """
        generated_images = []

        for i in range(num_images):
            v = (np.random.random_sample(self.input_size) < 0.5) * 1

            for j in range(gibbs_num_iter):
                p_h = self.input_output(v)
                h = (np.random.random_sample(self.output_size) < p_h) * 1
                p_h = self.output_input(h)
                v = (np.random.random_sample(self.input_size) < p_h) * 1

            generated_images.append(v.reshape(self.input_image_size))

        return generated_images
