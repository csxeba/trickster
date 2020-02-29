from typing import List

import tensorflow as tf

from .abstract import Callback
from ..utility.history import History


class OptimizerReset(Callback):

    def __init__(self, frequency: int, optimizers: List[tf.keras.optimizers.Optimizer], verbose: int = 1):
        super().__init__()
        self.frequency = frequency
        self.optimizers = optimizers
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, history: History = None):
        if epoch % self.frequency == 0:
            for opt in self.optimizers:
                for w in opt.weights:
                    w.assign(tf.zeros_like(w))
            print(f" [Trickster.OptimizerReset] - Resetted {len(self.optimizers)} optimizers.")
