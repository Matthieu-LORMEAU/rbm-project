import numpy as np

class RBM:
    """Create an RBM object
    """    
    def __init__(self, input_size, output_size):
        self.W = np.random.normal(0, 0.01, (input_size, output_size))
        self.a = np.zeros(input_size)
        self.b = np.zeros(output_size)
        
    def input_output(self, input):

    def output_input(self, output):

    def train(self, data, num_epochs, batch_size, lr):

    def generate_image(self,num_images, gibbs_num_iter):
