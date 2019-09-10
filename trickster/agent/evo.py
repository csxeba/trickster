import numpy as np
import keras

from ..utility import tensoric
from ..rollout import Trajectory


class EvolutionStrategies:

    def __init__(self, rollout: Trajectory, initial_stdev=1., population_size=100):
        self.stdev = initial_stdev
        self.reshaper = tensoric.TensorReshaper(self.model.get_weights())
        if not isinstance(self.stdev, np.ndarray):
            self.stdev = np.full(self.reshaper.N, self.stdev)

    def sample(self, state, reward, done):
        pass

    def push_experience(self, state, reward, done):
        pass
