import unittest

import numpy as np
from matplotlib import pyplot as plt

from trickster.processing import reward_shaping


class TestNumericDiscounters(unittest.TestCase):

    """Value: sum of future reward from given state"""

    def setUp(self):
        self.rwds = np.zeros(100)
        self.dones = np.zeros_like(self.rwds)
        self.dones[-1] = 1.

    def test_discounting(self):
        GAMMA = 0.99
        LAMBDA = 0.97
        shaper = reward_shaping.RewardShaper(GAMMA, LAMBDA)

        drwd = shaper.discount(self.rwds[1:], self.dones[1:], gamma=GAMMA)
        values = np.concatenate([np.array([drwd[0]]), drwd], axis=0)
        returns, advantages = shaper.compute_gae(self.rwds[1:], values[:-1], values[1:], self.dones[1:])
        gae_returns = advantages + values[1:]
        plt.figure(figsize=(16, 9))
        plt.plot(self.rwds[1:], "r-", label="Rewards", alpha=0.7)
        plt.plot(drwd, "g-", label="Discount", alpha=0.7)
        plt.plot(gae_returns, "b-", label="GAE", alpha=0.7)
        plt.xticks(np.arange(0, 100, step=10))
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()
