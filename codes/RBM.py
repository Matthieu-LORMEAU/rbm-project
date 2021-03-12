import numpy as np
from scipy.special import expit
import math
from codes.utils import reconstruction_error, sigmoid
from tqdm import tqdm

class RBM:
    """Create an RBM object."""

    def __init__(self, input_size: int, output_size: int):
        """Instantiate an RBM

        Parameters
        ----------
        input_size : int
            number of input features
        output_size : int
            number of hidden units
        """
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.normal(0, 0.01, (self.input_size, output_size))
        self.a = np.zeros(self.input_size)
        self.b = np.zeros(output_size)

    def input_output(self, input):
        """Compute the output for a certain input

        Parameters
        ----------
        input : ndarray
            (number of samples, input_size)

        Returns
        -------
        ndarray:
            (number of samples, output_size)
        """
        return sigmoid(input, self.W, self.b)

    def output_input(self, output):
        """Compute the input for a certain output

        Parameters
        ----------
        output : ndarray
            (number of samples, output_size)

        Returns
        -------
        ndarray
            (number of samples, input_size)
        """
        # shape data number * inputsize
        return sigmoid(output, self.W.T, self.a)

    def train(self, X, batch_size, num_epochs=100, lr=0.1, verbose=True):
        """Train the RBM object

        Parameters
        ----------
        X : ndarray
            (number of samples, input_size)
        batch_size : int
        num_epochs : int, optional
            number of epochs, by default 100
        lr : float, optional
            learning rate, by default 0.1
        errors : bool,
            to return and print the reconstruction score

        Returns
        -------
        list[float]
            List of reconstruction errors for each epoch

        Raises
        ------
        ValueError
            Dimension Error
        """
        if len(X.shape) != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                "Input dimensions must be (n_samples, RBM.input_size)")

        n_samples = X.shape[0]
        errors = []
        init_X = X.copy()

        tq_epochs = tqdm(range(num_epochs), leave=False)
        for e in tq_epochs:
            tq_epochs.set_description(f"Layer pretrain epoch : {e}")
            # shuffle data
            X = X[np.random.permutation(n_samples), :]
            for b in range(int(np.ceil(n_samples / batch_size))):
                # batch borders
                beg = b * batch_size
                end = min(beg + batch_size, n_samples)
                n = end - beg
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
            if verbose:
                error = 0
                for x in init_X:
                    p_h = self.input_output(x)
                    h = (np.random.random_sample(self.output_size) < p_h) * 1
                    p_X_tild = self.output_input(h)
                    x_tild = (np.random.random_sample(self.input_size) < p_X_tild) * 1
                    error += reconstruction_error(x, x_tild)
                errors.append(error/init_X.shape[0])
                if (e+1) % 100 == 0 :
                    epoch_stats = "Epoch {} Complete: Reconstruction Error: {:.7f}".format(
                        e + 1, errors[-1])
                    print(epoch_stats)

        if verbose:
            return errors

    def generate_image(self, num_images, gibbs_num_iter, reshape=None):
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
                p_h = self.input_output(v)
                h = (np.random.random_sample(self.output_size) < p_h) * 1
                p_v = self.output_input(h)
                v = (np.random.random_sample(self.input_size) < p_v) * 1

            generated_images.append(
                v if reshape is None else v.reshape(reshape))

        return generated_images
