from typing import Dict

import numpy as np


class History:

    def __init__(self, *keys):
        self.keys = keys
        self._logs = {key: [] for key in keys}
        self._buffer = {key: [] for key in keys}

    def record(self, **kwargs):
        for key, val in kwargs.items():
            self._logs[key].append(val)

    def buffer(self, **kwargs):
        for key, val in kwargs.items():
            self._buffer[key].append(val)

    def push_buffer(self):
        self.record(**{k: np.mean(v) for k, v in self._buffer.items()})
        self._buffer = {k: [] for k in self.keys}

    def gather(self):
        return self._logs

    def print(self,
              average_last=10,
              templates: Dict[str, str]=None,
              return_carriege=True,
              prefix=""):

        templates = templates or {}
        template = " ".join(templates.get(key, "{}") for key in self.keys)

        if average_last >= 2:
            values = [np.mean(self._logs[key][-average_last:]) for key in self.keys]
        else:
            values = [self._logs[key][-1] for key in self.keys]

        prefix = ("\r" if return_carriege else "") + prefix
        end = "" if return_carriege else None

        print(prefix + template.format(*values), end=end)
