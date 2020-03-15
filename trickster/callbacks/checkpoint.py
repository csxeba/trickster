import pathlib

import tensorflow as tf

from . import abstract
from ..utility.history import History
from ..utility import persistance


class ModelCheckpoint(abstract.Callback):

    def __init__(self,
                 checkpoint_dir: str,
                 name_suffix: str,
                 optimizer: tf.keras.optimizers.Optimizer = None,
                 frequency: int = 100,
                 weights_only: bool = False):

        super().__init__()
        self.path = pathlib.Path(checkpoint_dir)
        self.name_suffix = name_suffix
        self.optimizer = optimizer
        self.frequency = frequency
        self.weights_only = weights_only
        self.path.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, history: History = None):
        if epoch % self.frequency != 0:
            return
        data = history.last()
        sfx = self.name_suffix.format(epoch=epoch, **data)
        persistance.save_agent(self.rollout.agent, str(self.path), sfx)
