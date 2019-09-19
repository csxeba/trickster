import numpy as np


class Experience:

    def __init__(self, max_length=10000):
        self.memoirs = None
        self.max_length = max_length
        if not isinstance(self.max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer!")
        self._exclude_from_sampling = set()  # type: set
        self.final_state = None

    @staticmethod
    def _sanitize(args):
        Ns = set(list(map(len, args)))
        if len(Ns) != 1:
            raise ValueError("All arrays passed to remember() must be the same lenght!")

    def _initialize(self, num_memoirs):
        self.memoirs = [np.array([]) for _ in range(num_memoirs)]

    def reset(self):
        if self.memoirs is None:
            return
        self.memoirs = [np.array([]) for _ in range(self.width)]

    def _remember(self, arg, i):
        arg = np.array(arg, copy=False)
        if not self.memoirs[i].size:
            self.memoirs[i] = arg[-self.max_length:]
            return
        new = np.concatenate((self.memoirs[i][-self.max_length:], arg))
        self.memoirs[i] = new

    def remember(self, states, next_states, *args):
        args = (states, next_states) + args
        self._sanitize(args)
        if self.memoirs is None:
            self._initialize(num_memoirs=len(args))
        for i, arg in enumerate(args):
            self._remember(arg, i)

    def get_valid_indices(self):
        return list(range(0, self.N))

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
