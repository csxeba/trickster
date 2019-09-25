import unittest

import numpy as np
from matplotlib import pyplot as plt

from trickster.utility import numeric


class TestNumericDiscounters(unittest.TestCase):

    """Value: sum of future reward from given state"""

    def setUp(self):
        self.rwds = np.zeros(100)
        self.rwds[-2:] = 1.
        self.rwds[51] = 1.
        self.values = np.empty_like(self.rwds)
        self.values[:51] = 3.
        self.values[51:-2] = 2.
        self.values[-2] = 1.
        self.values[-1] = 0.
        self.dones = np.zeros_like(self.rwds)
        self.dones[-1] = 1.

    def test_discounting(self):
        GAMMA = 0.99
        LAMBDA = 0.95
        drwd = numeric.discount(self.rwds[1:], self.dones[1:], gamma=GAMMA)
        gae = numeric.compute_gae(self.rwds[1:], self.values[:-1], self.values[1:], self.dones[1:], GAMMA, LAMBDA)
        plt.figure(figsize=(16, 9))
        plt.plot(self.rwds[1:], "r-", label="Rewards")
        plt.plot(drwd, "g-", label="DRwd")
        plt.plot(gae, "b-", label="GAE")
        plt.plot(gae2, "y-", label="GAE2")
        plt.xticks(np.arange(0, 100, step=10))
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()
