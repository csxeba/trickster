from typing import List

from ..rollout import abstract
from ..utility.history import History

__all__ = ["Callback", "CallbackList"]


class Callback:

    def __init__(self):
        # noinspection PyTypeChecker
        self.rollout: abstract.RolloutInterface = None

    def set_rollout(self, rollout: abstract.RolloutInterface):
        self.rollout = rollout

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_begin(self, epoch: int, history: History = None):
        pass

    def on_epoch_end(self, epoch: int, history: History = None):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self, history: History):
        pass


class CallbackList(Callback):

    def __init__(self, callbacks: List[Callback]):
        super().__init__()
        self.callbacks = callbacks

    def set_rollout(self, rollout: abstract.RolloutInterface):
        for cb in self.callbacks:
            cb.set_rollout(rollout)

    def on_batch_begin(self):
        for cb in self.callbacks:
            cb.on_batch_begin()

    def on_batch_end(self):
        for cb in self.callbacks:
            cb.on_batch_end()

    def on_epoch_begin(self, epoch: int, history: History = None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, history)

    def on_epoch_end(self, epoch: int, history: History = None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, history)

    def on_train_begin(self):
        for cb in self.callbacks:
            cb.on_train_begin()

    def on_train_end(self, history: History):
        for cb in self.callbacks:
            cb.on_train_end(history)
