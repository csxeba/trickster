from collections import OrderedDict
from typing import Dict
import string

import numpy as np


class History:

    def __init__(self, *keys):
        self.keys = keys
        self._logs = {key: [] for key in keys}
        self._buffer = {key: [] for key in keys}
        self._strwidths = {key: max(len(key)+2, 11) for key in keys}

    def record(self, **kwargs):
        for key, val in kwargs.items():
            if key in self._logs:
                self._logs[key].append(val)

    def buffer(self, **kwargs):
        for key, val in kwargs.items():
            self._buffer[key].append(val)

    def push_buffer(self):
        new_record = {}
        for k, v in self._buffer.items():
            if len(v):
                new_record[k] = np.mean(v)
        self.record(**new_record)
        self._buffer = {k: [] for k in self.keys}

    def gather(self):
        return self._logs

    def last(self):
        result = OrderedDict()
        for k in self.keys:
            if len(self._logs[k]):
                result[k] = self._logs[k][-1]
        return result

    def print(self,
              average_last=10,
              return_carriege=True):

        if average_last >= 2:
            values = [np.mean(self._logs[key][-average_last:]) for key in self.keys]
        else:
            values = [self._logs[key][-1] for key in self.keys]

        prefix = "{:^7} | ".format(len(self))
        prefix = ("\r" if return_carriege else "") + prefix
        if prefix[-1] not in string.whitespace:
            prefix += " "
        end = "" if return_carriege else None

        logstrs = []
        for value, key in zip(values, self._strwidths):
            logstrs.append("{: ^{w}.4f}".format(value, w=self._strwidths[key]+1))
        logstr = prefix + "|".join(logstrs)

        print(logstr, end=end)

    def print_header(self):
        logstr = "|".join("{:^{w}}".format(key, w=self._strwidths[key]+1) for key in self.keys)
        logstr = "Episode | " + logstr
        print(logstr)
        print("-"*len(logstr))

    def __len__(self):
        return len(self._logs[self.keys[0]])
