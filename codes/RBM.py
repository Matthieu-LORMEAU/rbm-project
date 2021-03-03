import numpy as np
from scipy.special import expit


class RBM:
    """Create an RBM object"""

    def __init__(self, input_size, output_size):
        self.W = np.random.normal(0, 0.01, (input_size, output_size))
        self.a = np.zeros(input_size)
        self.b = np.zeros(output_size)
        self.input_size = input_size
        self.output_size = output_size

    def input_output(self, input):
        return expit(input @ self.W + self.b)

    def output_input(self, output):
        return expit(output @ self.W.T + self.a)

    def train(self, X, num_epochs, batch_size, lr):

        if len(X.shape) != 2 or X.shape[1] != self.input_size:
            raise ValueError("Input dimensions must be (n_samples, RBM.input_size)")

        n_samples = X.shape[0]

        for e in range(num_epochs):
            X = X[np.random.permutation(n_samples), :]

            for b in range(np.ceiling(n_samples / batch_size)):
                beg = b * batch_size
                end = np.min(beg + batch_size, n_samples)
                n = end - beg
                v0 = X[beg:end, :]
                prob_h0_v0 = self.input_output(v0)
                h0 = (np.random.random_sample((n, self.output_size)) < prob_h_batch) * 1
                prob_v1_h0 = self.output_input(h0)
                v1 = (np.random.random_sample((n, self.input_size)) < prob_v1_h0) * 1

    def generate_image(self, num_images, gibbs_num_iter):
        return
