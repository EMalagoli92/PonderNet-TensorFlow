import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='pondernet')
class KLDiv(tf.keras.losses.Loss):
    '''
    The Kullback-Leibler divergence loss with 'batchmean' reduction.
    '''
    def __call__(self,
                 y_true: tf.Tensor,
                 y_pred: tf.Tensor
                 ) -> tf.Tensor:
        batch_size = tf.shape(y_true)[0]
        batch_size = tf.cast(batch_size, dtype = tf.keras.backend.floatx())
        return tf.math.reduce_sum(y_pred * (tf.math.log(y_pred) - y_true)) / batch_size