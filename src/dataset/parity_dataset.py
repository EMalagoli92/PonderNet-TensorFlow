import tensorflow as tf
import numpy as np
from typing import Tuple


class ParityDataset(tf.keras.utils.Sequence):
    def __init__(self, 
                 n_samples: int, 
                 n_elems: int = 64, 
                 batch_size: int = 128
                 ):
        '''
        Parameters
        ----------
        n_samples : int
            Number of samples.
        n_elems : int, optional
            Number of elements in the input vector. 
            The default is 64.
        batch_size : int, optional
            Batch size. 
            The default is 128.
        '''
        self.n_samples = n_samples
        self.n_elems = n_elems
        self.batch_size = batch_size

    def __len__(self) -> int:
        return int(np.floor(self.n_samples) / self.batch_size)

    def __batch_generation(self) -> Tuple[tf.Tensor,tf.Tensor]:
        X = np.empty((self.batch_size, self.n_elems))
        Y = np.empty((self.batch_size))
        for i in range(self.batch_size):
            X[i] = np.zeros((self.n_elems))
            n_non_zero = np.random.randint(1, self.n_elems + 1, (1,))[0]
            X[i, :n_non_zero] = np.random.randint(0, 2, (n_non_zero,))*2 - 1
            X[i] = np.random.permutation(X[i])
            Y[i] = (X[i] == 1.).sum() % 2
        X = tf.convert_to_tensor(X, dtype=tf.keras.backend.floatx())
        Y = tf.convert_to_tensor(Y, dtype=tf.keras.backend.floatx())
        return X, Y

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_X, batch_Y = self.__batch_generation()
        return batch_X, batch_Y