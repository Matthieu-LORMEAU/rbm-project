from scipy.io import loadmat
import requests
import os

def load_alpha_digits():
    
    if (os.path.isdir('data')==0):
        os.mkdir('data')

    if (os.path.isdir('data/alpha_digits')==0):
        os.mkdir('data/alpha_digits')

    if (os.path.isfile('data/alpha_digits/binaryalphadigs.mat') == 0):
        r = requests.get("https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat")
        open('data/alpha_digits/binaryalphadigs.mat', 'wb').write(r.content)

    return loadmat('data/alpha_digits/binaryalphadigs.mat')

    
print(load_alpha_digits())

def load_mnist():
    if (os.path.isdir('data/mnist')==0):
        os.mkdir('data/mnist')
