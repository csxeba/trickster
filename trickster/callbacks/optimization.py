from typing import Dict, List

import numpy as np
import tensorflow as tf

from .abstract import Callback
from ..utility.history import History
from ..utility import model_utils


__all__ = ["OptimizerReset", "LearningRateScheduler"]


class OptimizerReset(Callback):

    def __init__(self,
                 frequency: int,
                 optimizer: tf.keras.optimizers.Optimizer,
                 verbose: int = 1):

        super().__init__()
        self.frequency = frequency
        self.optimizer = optimizer
        self.verbose = verbose

    @classmethod
    def make_many(cls,
                  frequency: int,
                  optimizers: List[tf.keras.optimizers.Optimizer],
                  verbose: int = 0):

        return [cls(frequency, opt, verbose) for opt in optimizers]

    def on_epoch_begin(self, epoch: int, history: History = None):
        if epoch % self.frequency == 0:
            model_utils.reset_optimizer(self.optimizer)


class LearningRateScheduler(Callback):

    def __init__(self,
                 learning_rates_per_epoch: np.ndarray,
                 optimizer: tf.keras.optimizers.Optimizer,
                 verbose: int,
                 wrap_around: bool = True,
                 reset_optimizer_on_new_cycle: bool = True):

        super().__init__()
        self.learning_rates = learning_rates_per_epoch
        self.optimizer = optimizer
        self.verbose = verbose
        self.wrap_around = wrap_around
        self.reset_optimizer = reset_optimizer_on_new_cycle
        self.cycles = 0
        self.lr = 0

    @classmethod
    def make_exponential(cls,
                         optimizer: tf.keras.optimizers.Optimizer,
                         num_epochs: int,
                         start_value: float,
                         decay_rate: float,
                         min_value: float = -np.inf,
                         verbose: int = 0,
                         wrap_around: bool = True,
                         reset_optimizer_on_new_cycle: bool = True):

        rates = np.empty(num_epochs, dtype="float32")
        current = start_value
        for i in range(num_epochs):
            rates[i] = current
            current *= decay_rate
            current = max(min_value, current)
        return cls(rates, optimizer, verbose, wrap_around, reset_optimizer_on_new_cycle)

    @classmethod
    def make_step(cls,
                  optimizer: tf.keras.optimizers.Optimizer,
                  num_epochs: int,
                  start_value: float,
                  epoch_steps: Dict[int, float],
                  verbose: int = 1,
                  wrap_around: bool = True,
                  reset_optimizer_on_new_cycle: bool = True):

        rates = np.full(num_epochs, fill_value=start_value, dtype="float32")
        for epoch in sorted(epoch_steps):
            rates[epoch:] = epoch_steps[epoch]
        return cls(rates, optimizer, verbose, wrap_around, reset_optimizer_on_new_cycle)

    def on_epoch_begin(self, epoch: int, history: History = None):
        if not self.wrap_around and epoch >= len(self.learning_rates):
            raise RuntimeError("Epoch overindexes schedule!")

        remainder = epoch % len(self.learning_rates)
        if remainder == 0:
            self.cycles += 1
            if self.cycles > 1 and self.reset_optimizer:
                model_utils.reset_optimizer(self.optimizer)

        self.lr = self.learning_rates[remainder]
        self.optimizer.learning_rate = self.lr
        if self.verbose:
            print(f" [Trickster.LearningRateScheduler] -"
                  f" Cycle {self.cycles} learning rate {self.lr}")
