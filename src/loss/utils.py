import tensorflow as tf


class KLDiv(tf.keras.losses.Loss):
    '''
    The Kullback-Leibler divergence loss with 'batchmean' reduction.
    '''
    def __call__(self,y_true: tf.Tensor,y_pred: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(y_true,out_type = tf.keras.backend.floatx())[0]
        return tf.math.reduce_sum(y_pred * (tf.math.log(y_pred) - y_true)) / batch