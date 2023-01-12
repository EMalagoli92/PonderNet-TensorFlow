import tensorflow as tf

from pondernet_tensorflow.loss.utils import KLDiv


@tf.keras.utils.register_keras_serializable(package="pondernet")
class RegularizationLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_p: float, max_steps: int = 1000, **kwargs):
        """
        Parameters
        ----------
        lambda_p : float
            The success probability of geometric distribution.
        max_steps : int, optional
            Highest N.
            The default is 1000.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.lambda_p = lambda_p
        self.max_steps = max_steps

        not_halted = 1.0
        p_g_list = []
        for k in range(self.max_steps):
            p_g_list.append(not_halted * self.lambda_p)
            not_halted = not_halted * (1 - self.lambda_p)
        p_g = tf.stack(p_g_list)
        self.p_g = tf.cast(
            tf.Variable(p_g, trainable=False, name="p_g"),
            dtype=tf.keras.backend.floatx(),
        )
        self.kl_div = KLDiv()

    def __call__(self, p: tf.Tensor):
        p = tf.transpose(p, perm=[1, 0])
        p_g = tf.broadcast_to(self.p_g[None, : tf.shape(p)[1]], shape=tf.shape(p))
        return self.kl_div(tf.math.log(p), p_g)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_p": self.lambda_p, "max_steps": self.max_steps})
        return config
