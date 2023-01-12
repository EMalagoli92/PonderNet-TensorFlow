__author__ = "EMalagoli92"
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

from .dataset.parity_dataset import ParityDataset
from .experiment import main
from .loss.reconstruction_loss import ReconstructionLoss
from .loss.regularization_loss import RegularizationLoss
from .models.parity_pondergru import ParityPonderGru

# Set Seed
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random_seed.set_seed(SEED)
np.random.seed(SEED)
