from collections import defaultdict

import numpy as np


class History:

    def __init__(self):
        self._logs = defaultdict(list)
        self._buffer = defaultdict(list)
        self._a_key = None

    def append(self, **kwargs):
        for key, val in kwargs.items():
            self._logs[key].append(val)

    def extend(self, **kwargs):
        val = None
        for key, val in kwargs.items():
            self._logs[key].extend(val)

    def buffer(self, **kwargs):
        for key, val in kwargs.items():
            self._buffer[key].append(val)

    def push_buffer(self):
        new_record = {}
        for k, v in self._buffer.items():
            if len(v):
                new_record[k] = np.mean(v)
        self.append(**new_record)
        self._buffer = defaultdict(list)

    def gather(self):
        return self._logs

    def last(self):
        result = dict()
        for k, v in self._logs.items():
            if len(v) > 0:
                result[k] = self._logs[k][-1]
        return result

    def reduce(self):
        return {k: np.mean(v) if len(v) else "?" for k, v in self._logs.items()}

    def incorporate(self, other: "History", reduce=True):
        if reduce:
            data = other.reduce()
            self.append(**data)
        else:
            self.extend(**other._logs)

    def __getitem__(self, item):
        return self._logs[item]

    def __len__(self):
        if self._a_key is None:
            self._a_key = list(self._logs.keys())[0]
        return len(self._logs[0])
