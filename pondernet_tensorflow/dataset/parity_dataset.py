import math
from typing import Tuple

import tensorflow as tf


class ParityDataset(tf.keras.utils.Sequence):
    def __init__(self, n_samples: int, n_elems: int = 64, batch_size: int = 128):
        """
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
        """
        self.n_samples = n_samples
        self.n_elems = n_elems
        self.batch_size = batch_size

    def __len__(self) -> int:
        return int(math.floor(self.n_samples) / self.batch_size)

    @tf.function
    def __batch_generation(self) -> Tuple[tf.Tensor, tf.Tensor]:
        X = []
        Y = []
        for _ in range(self.batch_size):
            n_non_zero = tf.random.uniform((), 1, self.n_elems + 1, tf.int32)
            x = tf.random.uniform((n_non_zero,), 0, 2, tf.int32) * 2 - 1
            x = tf.concat(
                [x, tf.zeros((self.n_elems - n_non_zero), dtype=tf.int32)], axis=0
            )
            x = tf.random.shuffle(x)
            y = tf.math.reduce_sum(tf.cast(tf.equal(x, 1), tf.int32)) % 2
            X.append(x)
            Y.append(y)
        X = tf.cast(tf.stack(X), tf.keras.backend.floatx())
        Y = tf.cast(tf.stack(Y), tf.keras.backend.floatx())
        return X, Y

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_X, batch_Y = self.__batch_generation()
        return batch_X, batch_Y
