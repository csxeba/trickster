import unittest

import numpy as np
from matplotlib import pyplot as plt

from trickster.utility import advantage_utils


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
        LAMBDA = 0.97
        drwd = advantage_utils.discount(self.rwds[1:], self.dones[1:], gamma=GAMMA)
        advantages = advantage_utils.compute_gae(self.rwds[1:], self.values[:-1], self.values[1:], self.dones[1:], GAMMA, LAMBDA)
        gae_returns = advantages + self.values[:-1]
        plt.figure(figsize=(16, 9))
        plt.plot(self.rwds[1:], "r-", label="Rewards", alpha=0.7)
        plt.plot(drwd, "g-", label="DRwd", alpha=0.7)
        plt.plot(gae_returns, "b-", label="GAE", alpha=0.7)
        plt.xticks(np.arange(0, 100, step=10))
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()
