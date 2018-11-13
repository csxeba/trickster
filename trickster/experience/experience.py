import numpy as np


class Experience:

    def __init__(self, max_length=10000):
        self.memoirs = None
        self.max_length = max_length

    @staticmethod
    def sanitize(args):
        N0 = len(args[0])
        if not all(N == N0 for N in map(len, args[1:])):
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

    def remember(self, *args):
        self.sanitize(args)
        if self.memoirs is None:
            self.reset(len(args))
        for i, arg in enumerate(args):
            self._remember(arg, i)

    def sample(self, size=32):
        if not self.N:
            return [[]] * len(self.memoirs)
        size = min(size, self.N)
        idx = np.random.randint(0, self.N, size=size)
        return [mem[idx] for mem in self.memoirs]

    def stream(self, size=32, infinite=False):
        N = len(self.memoirs[0])
        arg = np.arange(N)
        while 1:
            np.random.shuffle(arg)
            for start in range(0, len(arg), size):
                idx = arg[start:start+size]
                yield [mem[idx] for mem in self.memoirs]
            if not infinite:
                break

    @property
    def N(self):
        return len(self.memoirs[0])
