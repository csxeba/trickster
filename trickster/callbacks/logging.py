import os
from typing import List, Dict, Any

import tensorflow as tf
import numpy as np

from .abstract import Callback
from ..utility import visual, path_utils
from ..utility.history import History

__all__ = ["ProgressPrinter", "HistoryPlotter", "CSVLogger"]


class ProgressPrinter(Callback):

    def __init__(self,
                 keys: List[str],
                 formatters: Dict[str, str] = None,
                 prefix: str = "Epoch",
                 average_last: int = 10):

        super().__init__()
        self.keys = keys
        self.prefix = prefix
        self.strwidths = {key: max(len(key) + 2, 11) for key in [prefix] + keys}
        if formatters is None:
            keys = [prefix] + keys
            templates = ["{:>{w}}"] + ["{: ^{w}.4f}"] * len(self.keys)
            formatters = {key: template for key, template in zip(keys, templates)}
        self.formatters = formatters
        self.header = " | ".join("{:^{w}}".format(key, w=self.strwidths[key]) for key in [self.prefix] + self.keys)
        self.separator = "-" * len(self.header)
        self.average_last = average_last

    def on_epoch_begin(self, epoch: int, history: History = None):
        if (epoch-1) % (self.average_last*10) == 0:
            print()
            print(self.header)
            print(self.separator)

    def on_epoch_end(self, epoch: int, history: History = None):

        if self.average_last > 1:
            values: List[Any] = [np.mean(history[key][-self.average_last:]) for key in self.keys]
            for key in self.keys:
                data = history[key][-self.average_last:]
                if not data:
                    values.append("None")
                    continue
                values.append(np.mean(data))
        else:
            values = [history[key][-1] if history[key] else "None" for key in self.keys]

        keys = [self.prefix] + self.keys
        values = [epoch] + values

        logstrs = []
        for value, key in zip(values, keys):
            formatter = self.formatters[key]
            logstrs.append(formatter.format(value, w=self.strwidths[key]))
        logstr = " | ".join(logstrs)

        print("\r" + logstr, end="")
        if epoch % self.average_last == 0:
            print()


class HistoryPlotter(Callback):

    def __init__(self, smoothing_window_size: int = 10):
        super().__init__()
        self.smoothing_window_size = smoothing_window_size

    def on_train_end(self, history: History):
        visual.plot_history(history, smoothing_window_size=self.smoothing_window_size)


class CSVLogger(Callback):

    def __init__(self, path="default"):
        super().__init__()
        if path == "default":
            path = os.path.join(path_utils.defaults.logdir, "log.csv")
        self.path = path
        self.initialized = False
        print(" [Trickster.CSVLogger] - Logging to", self.path)

    def on_epoch_end(self, epoch: int, history: History = None):
        data = {k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in history.last().items()}
        if not self.initialized:
            line = ",".join(history.keys)
            self.initialized = True
        else:
            line = ""
        line = line + "\n"
        line = line + ",".join(str(data[key]) for key in history.keys)
        with open(self.path, "a") as handle:
            handle.write(line)
