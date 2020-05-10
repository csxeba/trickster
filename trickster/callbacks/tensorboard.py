import tensorflow as tf

from .abstract import Callback
from ..utility.history import History
from ..utility.artifactory import Artifactory


class TensorBoard(Callback):

    def __init__(self, artifactory: Artifactory = "default"):
        super().__init__()
        if artifactory == "default":
            artifactory = Artifactory.make_default()
        self.artifactory = artifactory
        self.writer = tf.summary.create_file_writer(str(artifactory.tensorboard))
        print(" [Trickster.TensorBoard] - Logdir:", artifactory.tensorboard)
        self.writer.set_as_default()

    def on_train_begin(self):
        tf.summary.trace_on(graph=True, profiler=True)

    def on_epoch_end(self, epoch: int, history: History = None):
        last = history.last()
        self.writer.set_as_default()
        for key, val in last.items():
            tf.summary.scalar(key, val, epoch)

    def on_train_end(self, history: History):
        self.writer.set_as_default()
        tf.summary.trace_export(self.rollout.experiment_name, step=0, profiler_outdir=self.artifactory.tensorboard)
