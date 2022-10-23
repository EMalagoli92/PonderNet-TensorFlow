import tensorflow as tf


class KLDiv(tf.keras.losses.Loss):
    def __call__(self,y_true: tf.Tensor,y_pred: tf.Tensor) -> tf.Tensor:
        batch = y_true.get_shape()[0]
        return tf.math.reduce_sum(y_pred * (tf.math.log(y_pred) - y_true)) / batch