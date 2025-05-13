import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class MNISTDataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.X, self.y = self.load_data()
        self.X_images = self.X.reshape(-1, 28, 28)

    def load_data(self):
        if not os.path.exists(f'{self.data_dir}/mnist_data.npy') or not os.path.exists(f'{self.data_dir}/mnist_target.npy'):
            # download the MNIST dataset if it does not exist
            # it may take a while
            mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False, data_home=self.data_dir)
            X, y = mnist["data"], mnist["target"]
            np.save(f'{self.data_dir}/mnist_data.npy', X)
            np.save(f'{self.data_dir}/mnist_target.npy', y)
        else:
            X = np.load(f'{self.data_dir}/mnist_data.npy', allow_pickle=True)
            y = np.load(f'{self.data_dir}/mnist_target.npy', allow_pickle=True)

        return X / 255, y.astype(np.int8)