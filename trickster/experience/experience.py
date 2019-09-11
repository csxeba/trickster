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
        self._exclude_from_sampling = set()  # type: set

    @staticmethod
    def sanitize(args):
        Ns = set(list(map(len, args)))
        if len(Ns) != 1:
            raise ValueError("All arrays passed to remember() must be the same lenght!")

    def initialize(self, num_memoirs):
        self.memoirs = [np.array([]) for _ in range(num_memoirs)]

    def reset(self):
        if self.memoirs is None:
            return
        self.memoirs = [np.array([]) for _ in range(self.width)]

    def _remember(self, arg, i):
        if not self.memoirs[i].size:
            self.memoirs[i] = arg[-self.max_length:]
            return
        new = np.concatenate((self.memoirs[i][-self.max_length:], arg))
        self.memoirs[i] = new

    def remember(self, states, *args, dones, exclude=()):
        args = (states,) + args + (dones,)
        self.sanitize(args)
        if self.memoirs is None:
            self.initialize(num_memoirs=len(args))
        num_new_memories = len(states)
        N_before_update = self.N
        N_after_update = N_before_update + num_new_memories
        num_dropped_memories = N_after_update - self.max_length

        for i, arg in enumerate(args):
            self._remember(arg, i)

        if any(dones):
            exclude_dones = (np.argwhere(dones)+1)
            exclude_dones = exclude_dones[:, 0]
            exclude = tuple(exclude_dones) + exclude

        if exclude:
            exclude = np.array(exclude)
            exclude[exclude < 0] = len(states) + exclude[exclude < 0]
            self._exclude_from_sampling.update(
                set(exclude + N_before_update)
            )

        if num_dropped_memories > 0:
            self._exclude_from_sampling = [i-num_dropped_memories for i in list(self._exclude_from_sampling)]
            self._exclude_from_sampling = {i for i in self._exclude_from_sampling if i >= 0}

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
