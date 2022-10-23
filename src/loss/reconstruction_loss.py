import tensorflow as tf


class ReconstructionLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 loss_func: tf.keras.losses,
                 **kwargs,
                 ):
        '''
        Parameters
        ----------
        loss_func : tf.keras.losses
            Loss function.
        '''
        super().__init__(**kwargs)
        self.loss_func = loss_func

    def __call__(self, 
                 p: tf.Tensor, 
                 y_hat: tf.Tensor, 
                 y: tf.Tensor
                 ) -> tf.Tensor:
        total_loss = tf.zeros((1), dtype=p.dtype)
        for n in range(tf.shape(p)[0]):
            loss = tf.math.reduce_mean(
                (p[n] * self.loss_func(y, y_hat[n])), keepdims=True)
            total_loss += loss
        return tf.squeeze(total_loss)