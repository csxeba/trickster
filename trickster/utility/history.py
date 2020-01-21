import numpy as np


class History:

    def __init__(self, *keys):
        self.keys = list(keys)
        self._logs = {key: [] for key in keys}
        self._buffer = {key: [] for key in keys}

    def append(self, **kwargs):
        for key, val in kwargs.items():
            if key in self._logs:
                self._logs[key].append(val)

    def extend(self, **kwargs):
        for key, val in kwargs.items():
            if key in self._logs:
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
        self._buffer = {k: [] for k in self.keys}

    def gather(self):
        return self._logs

    def last(self):
        result = dict()
        for k in self.keys:
            if len(self._logs[k]):
                result[k] = self._logs[k][-1]
        return result

    def reduce(self):
        return {k: np.mean(v) for k, v in self._logs.items()}

    def incorporate(self, other: "History", reduce=True):
        if reduce:
            data = other.reduce()
            self.append(**data)
        else:
            self.extend(**other._logs)

    def __getitem__(self, item):
        return self._logs[item]

    def __len__(self):
        return len(self._logs[self.keys[0]])
