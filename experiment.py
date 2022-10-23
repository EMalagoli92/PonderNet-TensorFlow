import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed
import time
from collections import defaultdict
from typing import Tuple
from src.dataset.parity_dataset import ParityDataset
from src.parity_pondergru import ParityPonderGru
from src.loss.reconstruction_loss import ReconstructionLoss
from src.loss.regularization_loss import RegularizationLoss

# Set Seed
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random_seed.set_seed(SEED)
np.random.seed(SEED)


tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float64')


class Configs(object):
    def __init__(self,
                 epochs: int = 100,
                 n_batches: int = 500,
                 batch_size: int = 128,
                 n_elems: int = 8,
                 n_hidden: int = 64,
                 max_steps: int = 20,
                 lambda_p: float = .2,
                 beta: float = .01,
                 grad_norm_clip: float = 1.0,
                 learning_rate: float = 3e-4
                 ):
        '''
        Train a PonderNet on Parity Task.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs. 
            The default is 100.
        n_batches : int, optional
            Number of batches per epoch. 
            The default is 500.
        batch_size : int, optional
            Batch size. 
            The default is 128.
        n_elems : int, optional
            The number of elements in the input vector.
            The default is 8.
        n_hidden : int, optional
            Number of units in the hidden layer (state).
            The default is 64.
        max_steps : int, optional
            Maximum number of steps N.
            The default is 20.
        lambda_p : float, optional
            Lambda for geometric distribution.
            The default is .2.
        beta : float, optional
            Regularization loss coefficient.
            The default is .01.
        grad_norm_clip : float, optional
            Gradient clipping by norm.
            The default is 1.0.
        learning_rate : float, optional
            Learning rate for Adam optimizer. 
            The default is 3e-4.
        '''
        self.epochs = epochs
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.beta = beta
        self.grad_norm_clip = grad_norm_clip
        self.learning_rate = learning_rate
        self.train_loader = ParityDataset(self.batch_size * self.n_batches, 
                                          self.n_elems, 
                                          self.batch_size)
        self.valid_loader = ParityDataset(self.batch_size * 32, 
                                          self.n_elems, 
                                          self.batch_size)
        self.model = ParityPonderGru(self.n_elems, 
                                     self.n_hidden, 
                                     self.max_steps)
        self.loss_rec = ReconstructionLoss(tf.nn.sigmoid_cross_entropy_with_logits)
        self.loss_reg = RegularizationLoss(self.lambda_p, self.max_steps)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate,epsilon=1e-08)
        self.train_acc_metric = tf.keras.metrics.Accuracy()
        self.val_acc_metric = tf.keras.metrics.Accuracy()
        self.history = defaultdict(list)
        
    def ProgBar(self):
        self.stateful_metrics =['train_loss', 
                                'train_loss_reg',
                                'train_steps',
                                'val_loss',
                                'val_loss_reg',
                                'train_acc',
                                'val_acc',
                                'val_steps'
                                ]
        self.ProgBar_ = tf.keras.utils.Progbar(self.n_batches, 
                                               stateful_metrics = self.stateful_metrics,
                                               unit_name = 'batch',
                                               width = 30
                                               )

    def history_update(self,values):
        for tuple in values: self.history[tuple[0]].append(tuple[1].numpy())

    @tf.function
    def train_step(self,
                   batch_X: tf.Tensor, 
                   batch_Y: tf.Tensor
                   ) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
        with tf.GradientTape() as tape:
            p, y_hat, p_sampled, y_hat_sampled = self.model(batch_X,training=True)
            loss_rec = self.loss_rec(p,y_hat, batch_Y)
            loss_reg = self.loss_reg(p)
            loss = loss_rec + self.beta * loss_reg
        grads = tape.gradient(loss, self.model.trainable_weights)
        total_norm = tf.linalg.global_norm(grads)
        grads = tf.clip_by_global_norm(grads, use_norm = total_norm + 1e-6,clip_norm=self.grad_norm_clip)[0]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y_hat_sampled >0, batch_Y)
        steps = tf.range(1,tf.shape(p)[0] +1,dtype=p.dtype)
        expected_steps = tf.reduce_sum(p * steps[:,None],axis=0)
        expected_steps = tf.reduce_mean(expected_steps)
        return loss_rec, loss_reg, expected_steps
    
    @tf.function
    def valid_step(self,
                   batch_X: tf.Tensor, 
                   batch_Y: tf.Tensor
                   ) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
        p, y_hat, p_sampled, y_hat_sampled = self.model(batch_X,training=False)
        loss_rec = self.loss_rec(p,y_hat, batch_Y)
        loss_reg = self.loss_reg(p)
        self.val_acc_metric.update_state(y_hat_sampled>0, batch_Y)
        steps = tf.range(1,tf.shape(p)[0] +1,dtype=p.dtype)
        expected_steps = tf.reduce_sum(p * steps[:,None],axis=0)
        expected_steps = tf.reduce_mean(expected_steps)
        return loss_rec, loss_reg, expected_steps
    
    def run(self) -> dict:
        start_time = time.time()
        for epoch in range(self.epochs):
            self.history['epochs'].append(epoch+1)
            print("\nepoch {}/{}".format(epoch+1,self.epochs))
            self.ProgBar()
            for step, (batch_X_train, batch_Y_train) in enumerate(self.train_loader):
                train_loss_rec, train_loss_reg, train_expected_steps = self.train_step(batch_X_train,batch_Y_train)
                values=[('train_loss',train_loss_rec),('train_loss_reg',train_loss_reg),('train_steps',train_expected_steps)]
                self.ProgBar_.update(step, values=values)
            train_acc = self.train_acc_metric.result()
            values += [('train_acc',train_acc)]
            self.ProgBar_.update(step, values=values) 
            self.train_acc_metric.reset_states()
            for batch_X_val, batch_Y_val in self.valid_loader:
                val_loss_rec, val_loss_reg, val_expected_steps = self.valid_step(batch_X_val, batch_Y_val)
            val_acc = self.val_acc_metric.result()
            values += [('val_loss',val_loss_rec),('val_loss_reg',val_loss_reg),('val_steps',val_expected_steps),('val_acc',val_acc)]
            self.ProgBar_.update(self.n_batches, values=values, finalize=True)
            self.val_acc_metric.reset_states()
            self.history_update(values)
        elapsed_time = time.time() - start_time
        print('\nTraining executed in: {:.3f} seconds'.format(float(elapsed_time)))
        return dict(self.history)


def main():
    experiment = Configs()
    return experiment.run()
            
        
if __name__ == '__main__':
    history = main()