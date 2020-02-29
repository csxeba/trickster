import tensorflow as tf

from .abstract import Callback
from ..utility import path_utils
from ..utility.history import History


class TensorBoard(Callback):

    def __init__(self, logdir="default"):
        super().__init__()
        if logdir == "default":
            logdir = path_utils.defaults.logdir
        self.writer = tf.summary.create_file_writer(logdir)
        print(" [Trickster.TensorBoard] - Logdir:", logdir)

    def on_epoch_end(self, epoch: int, history: History = None):
        last = history.last()
        self.writer.set_as_default()
        for key, val in last.items():
            tf.summary.scalar(key, val, epoch)
