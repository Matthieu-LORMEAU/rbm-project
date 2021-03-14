from scipy.io import loadmat
import requests
import os
import gzip
import numpy as np
from numba import njit
from config import ROOT_DIR


def reconstruction_error(X, X_tild):
    """Return the reconstruction error score for an array of inputs

        Parameters
        ----------
        X : ndarray
            (number of samples, input_size)
        X_tild : ndarray
            (number of samples, input_size)

        Returns
        -------
        float
            reconstruction error
        """
    return np.linalg.norm(X - X_tild)


def cross_entropy(predictions, targets):
    """Compute the Cross Entropy

    Parameters
    ----------
    predictions : ndarray
        (n_samples, n_classes)
    targets : ndarray
        (n_samples, n_classes)

    Returns
    -------
    Float
        number
    """
    likelihood = targets * np.log(predictions)
    return -np.sum(likelihood) / predictions.shape[0]


@njit(fastmath=True, parallel=True)
def sigmoid(X, W, b):
    """Sigmoid function

    Parameters
    ----------
    X : [type]
        [description]
    W : [type]
        [description]
    b : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return 1 / (1 + np.exp(-(np.dot(X.astype(np.float64), W) + b)))


def accuracy_score(predictions, targets):
    """Compute accuracy score for one hot encoded vectors

    Parameters
    ----------
    predictions : [type]
        [description]
    targets : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    if len(targets.shape) > 1:
        targets = np.argmax(targets, axis=1)

    return np.sum(predictions == targets) * 1 / predictions.shape[0]


def load_alpha_digits():
    os.chdir(ROOT_DIR)

    if (os.path.isdir('data') == 0):
        os.mkdir('data')

    if (os.path.isdir('data/alpha_digits') == 0):
        os.mkdir('data/alpha_digits')

    if (os.path.isfile('data/alpha_digits/binaryalphadigs.mat') == 0):
        r = requests.get("https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat")
        open('data/alpha_digits/binaryalphadigs.mat', 'wb').write(r.content)

    return loadmat('data/alpha_digits/binaryalphadigs.mat')


def load_mnist(train_data=True, test_data=False):
    """
    Get mnist data from the official website and
    load them in binary format.

    Parameters
    ----------
    train_data : bool
        Loads
        'train-images-idx3-ubyte.gz'
        'train-labels-idx1-ubyte.gz'
    test_data : bool
        Loads
        't10k-images-idx3-ubyte.gz'
        't10k-labels-idx1-ubyte.gz' 

    Return
    ------
    tuple
    tuple[0] are images (train & test)
    tuple[1] are labels (train & test)

    """
    os.chdir(ROOT_DIR)
    RESOURCES = [
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
    ]

    if (os.path.isdir('data') == 0):
        os.mkdir('data')
    if (os.path.isdir('data/mnist') == 0):
        os.mkdir('data/mnist')
    for name in RESOURCES:
        if (os.path.isfile('data/mnist/' + name) == 0):
            url = 'http://yann.lecun.com/exdb/mnist/' + name
            r = requests.get(url, allow_redirects=True)
            open('data/mnist/' + name, 'wb').write(r.content)

    return get_images(train_data, test_data), get_labels(train_data, test_data)


def get_images(train_data=True, test_data=False):

    to_return = []

    if train_data:
        with gzip.open('data/mnist/train-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            train_images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            to_return.append(np.where(train_images > 127, 1, 0))

    if test_data:
        with gzip.open('data/mnist/t10k-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            test_images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            to_return.append(np.where(test_images > 127, 1, 0))

    return to_return


def get_labels(train_data=True, test_data=False):

    to_return = []

    if train_data:
        with gzip.open('data/mnist/train-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            train_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(train_labels)
    if test_data:
        with gzip.open('data/mnist/t10k-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            test_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(test_labels)

    return to_return
