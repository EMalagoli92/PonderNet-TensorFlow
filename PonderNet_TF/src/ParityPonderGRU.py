from typing import Tuple
import tensorflow as tf
import tensorflow_probability as tfp
import math


class ClassSigmoid(object):
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.keras.activations.sigmoid(x)

class ParityPonderGru(tf.keras.models.Model):
    def __init__(self, n_elems: int, n_hidden: int, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.n_hidden = n_hidden

        limit = math.sqrt(1/n_hidden)
        initializer = tf.keras.initializers.RandomUniform(
            minval=-limit, maxval=limit)

        self.gru = tf.keras.layers.GRUCell(input_shape=(n_elems,),
                                           units=n_hidden,
                                           kernel_initializer=initializer,
                                           recurrent_initializer=initializer,
                                           bias_initializer=initializer)

        self.output_layer = tf.keras.layers.Dense(input_shape=(n_hidden,),
                                                  units=1,
                                                  kernel_initializer=initializer,
                                                  bias_initializer=initializer)

        self.lambda_layer = tf.keras.layers.Dense(input_shape=(n_hidden,),
                                                  units=1,
                                                  kernel_initializer=initializer,
                                                  bias_initializer=initializer)

        self.lambda_prob = ClassSigmoid()
        self.is_halt = False

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = x.get_shape()[0]
        h = tf.zeros((x.get_shape()[0], self.n_hidden), dtype=x.dtype)
        _, h = self.gru(x, h)

        p = []
        y = []

        un_halted_prob = tf.ones((batch_size,), dtype=h.dtype)
        halted = tf.zeros((batch_size,), dtype=h.dtype)
        p_m = tf.zeros((batch_size,), dtype=h.dtype)
        y_m = tf.zeros((batch_size,), dtype=h.dtype)

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = tf.ones(h.get_shape()[0], dtype=h.dtype)
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]

            y_n = self.output_layer(h)[:, 0]
            p_n = un_halted_prob * lambda_n
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            halt = tfp.distributions.Bernoulli(
                probs=lambda_n, dtype=tf.float32).sample() * (1 - halted)

            p.append(p_n)
            y.append(y_n)

            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt

            halted = halted + halt

            _, h = self.gru(x, h)

            if self.is_halt and tf.reduce_sum(halted).numpy() == batch_size:
                break
        return tf.stack(p), tf.stack(y), p_m, y_m