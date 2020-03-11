from typing import List

import numpy as np
import tensorflow as tf

from .abstract import Callback
from ..utility.history import History


class OptimizerReset(Callback):

    def __init__(self, frequency: int, optimizer: tf.keras.optimizers.Optimizer, verbose: int = 1):
        super().__init__()
        self.frequency = frequency
        self.optimizer = optimizer
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, history: History = None):
        if epoch % self.frequency == 0:
            for w in self.optimizer.weights:
                w.assign(tf.zeros_like(w))


class LearningRateScheduler(Callback):

    def __init__(self,
                 learning_rates_per_epoch: np.ndarray,
                 optimizer: tf.keras.Model,
                 verbose: int):

        super().__init__()
        self.learning_rates = learning_rates_per_epoch
        self.optimizer = optimizer
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, history: History = None):
        new_learning_rate = self.learning_rates[epoch]
        self.optimizer.learning_rate = new_learning_rate
        if self.verbose:
            print(f" [Trickster.LearningRateScheduler] - Learning rate set to {new_learning_rate}")