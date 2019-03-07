import numpy as np


class Experience:

    def __init__(self, max_length=10000):
        self.memoirs = None
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = 0
        if not isinstance(self.max_length, int):
            raise ValueError("max_length must be an integer!")
        if self.max_length < 0:
            self.max_length = 0
        self._exclude_from_sampling = set()

    @staticmethod
    def sanitize(args):
        Ns = set(list(map(len, args)))
        if len(Ns) != 1:
            raise ValueError("All arrays passed to remember() must be the same lenght!")

    def reset(self, num_memoirs=None):
        if num_memoirs is None:
            num_memoirs = len(self.memoirs)
        if num_memoirs < 1:
            raise ValueError("Invalid number of memoirs specified")
        self.memoirs = [np.array([]) for _ in range(num_memoirs)]

    def _remember(self, arg, i):
        if not self.memoirs[i].size:
            self.memoirs[i] = arg[-self.max_length:]
            return
        new = np.concatenate((self.memoirs[i][-self.max_length:], arg))
        self.memoirs[i] = new

    def remember(self, states, *args, exclude=()):
        args = (states,) + args
        self.sanitize(args)
        if self.memoirs is None:
            self.reset(len(args))
        N_before_update = self.N
        for i, arg in enumerate(args):
            self._remember(arg, i)
        if exclude:
            exclude = np.array(exclude)
            exclude[exclude < 0] = len(states) + exclude[exclude < 0]
            self._exclude_from_sampling.update(
                set(exclude + N_before_update)
            )

    def get_valid_indices(self):
        return [i for i in range(0, self.N-1) if i not in self._exclude_from_sampling]

    @property
    def N(self):
        if self.memoirs is None:
            return 0
        return len(self.memoirs[0])

    @property
    def width(self):
        if self.memoirs is None:
            return 0
        return len(self.memoirs)
