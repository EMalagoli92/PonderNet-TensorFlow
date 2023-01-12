import math

import tensorflow as tf
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(package="pondernet")
class ParityPonderGru(tf.keras.Model):
    def __init__(self, n_elems: int, n_hidden: int, max_steps: int, **kwargs):
        """
        Parameters
        ----------
        n_elems : int
            Number of elements in the input vector.
        n_hidden : int
            The state vector size of the GRU.
        max_steps : int
            Maximum number of steps.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps

        limit = math.sqrt(1 / self.n_hidden)
        initializer = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        self.gru = tf.keras.layers.GRUCell(
            input_shape=(self.n_elems,),
            units=self.n_hidden,
            kernel_initializer=initializer,
            recurrent_initializer=initializer,
            bias_initializer=initializer,
            name="gru",
        )
        self.output_layer = tf.keras.layers.Dense(
            input_shape=(self.n_hidden,),
            units=1,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            name="output_layer",
        )
        self.lambda_layer = tf.keras.layers.Dense(
            input_shape=(self.n_hidden,),
            units=1,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            name="lambda_layer",
        )
        self.lambda_prob = tf.keras.layers.Activation(tf.nn.sigmoid, name="lambda_prob")

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        h = tf.zeros((batch_size, self.n_hidden), dtype=inputs.dtype)
        _, h = self.gru(inputs, h)

        p = []
        y = []

        un_halted_prob = tf.ones((batch_size,), dtype=h.dtype)
        halted = tf.zeros((batch_size,), dtype=h.dtype)
        p_m = tf.zeros((batch_size,), dtype=h.dtype)
        y_m = tf.zeros((batch_size,), dtype=h.dtype)

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = tf.ones(tf.shape(h)[0], dtype=h.dtype)
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]

            y_n = self.output_layer(h)[:, 0]
            p_n = un_halted_prob * lambda_n
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            halt = tfp.distributions.Bernoulli(
                probs=lambda_n, dtype=p_m.dtype
            ).sample() * (1 - halted)

            p.append(p_n)
            y.append(y_n)

            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt

            halted = halted + halt

            _, h = self.gru(inputs, h)

        return tf.stack(p), tf.stack(y), p_m, y_m

    def build(self, input_shape):
        super().build(input_shape)

    def __to_functional(self):
        if self.built:
            x = tf.keras.layers.Input(shape=(self._build_input_shape[1:]))
            model = tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)
        else:
            raise ValueError(
                "This model has not yet been built. "
                "Build the model first by calling build() or "
                "by calling the model on a batch of data."
            )
        return model

    def summary(self, *args, **kwargs):
        self.__to_functional()
        super().summary(*args, **kwargs)

    def plot_model(self, *args, **kwargs):
        tf.keras.utils.plot_model(model=self.__to_functional(), *args, **kwargs)

    def save(self, *args, **kwargs):
        self.__to_functional().save(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_elems": self.n_elems,
                "n_hidden": self.n_hidden,
                "max_steps": self.max_steps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
