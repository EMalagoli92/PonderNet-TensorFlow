import tensorflow as tf


class KLDiv(object):
    def __call__(self,y_true: tf.Tensor,y_pred: tf.Tensor) -> tf.Tensor:
        batch = y_true.get_shape()[0]
        return tf.math.reduce_sum(y_pred * (tf.math.log(y_pred) - y_true)) / batch

class RegularizationLoss(object):
    def __init__(self,lambda_p: float, max_steps: int = 1_000):
        super().__init__()
        not_halted = 1.0
        p_g_list = []
        for k in range(max_steps):
            p_g_list.append(not_halted * lambda_p)
            not_halted = not_halted * (1-lambda_p)
        p_g = tf.stack(p_g_list)
        self.p_g = tf.cast(tf.Variable(p_g,trainable=False,name='p_g'),dtype=tf.float64)
        self.kl_div = KLDiv()

    def __call__(self,p: tf.Tensor) -> tf.Tensor:
        p = tf.transpose(p,perm=[1,0])
        p_g = tf.broadcast_to(self.p_g[None,:p.get_shape()[1]],shape = p.get_shape())
        return self.kl_div(tf.math.log(p), p_g)