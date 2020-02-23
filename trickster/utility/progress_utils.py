from typing import List, Dict

import numpy as np

from .history import History


class ProgressPrinter:

    def __init__(self,
                 keys: List[str],
                 formatters: Dict[str, str] = None,
                 prefix: str = "Epoch"):

        self.keys = keys
        self.prefix = prefix
        self.strwidths = {key: max(len(key)+2, 11) for key in [prefix] + keys}
        if formatters is None:
            keys = [prefix] + keys
            templates = ["{:>{w}}"] + ["{: ^{w}.4f}"] * len(self.keys)
            formatters = {key: template for key, template in zip(keys, templates)}
        self.formatters = formatters
        self.header = " | ".join("{:^{w}}".format(key, w=self.strwidths[key]) for key in [self.prefix] + self.keys)
        self.separator = "-"*len(self.header)

    def print(self,
              history: History,
              average_last=10,
              return_carriege=True):

        if average_last > 1:
            values = [np.mean(history[key][-average_last:]) for key in self.keys]
            for key in self.keys:
                data = history[key][-average_last:]
                if not data:
                    values.append("None")
                    continue
                values.append(np.mean(data))
        else:
            values = [history[key][-1] if history[key] else "None" for key in self.keys]

        keys = [self.prefix] + self.keys
        values = [len(history)] + values

        logstrs = []
        for value, key in zip(values, keys):
            formatter = self.formatters[key]
            logstrs.append(formatter.format(value, w=self.strwidths[key]))
        logstr = " | ".join(logstrs)

        end = "" if return_carriege else None
        prefix = "\r" if return_carriege else ""

        print(prefix + logstr, end=end)

    def print_header(self):
        print(self.header)
        print(self.separator)

    def override_formatter(self, key, template):
        self.formatters[key] = template
