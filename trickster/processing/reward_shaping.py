from typing import Union

import numpy as np

from ..utility import numeric_utils


class RewardShaper:

    def __init__(self,
                 discount_factor_gamma: float = 0.99,
                 gae_lambda: Union[float, None] = 0.97):

        self.gamma = discount_factor_gamma
        self.lmbda = gae_lambda

    def discount(self, rewards, dones, gamma=None, normalize=None):
        if gamma is None:
            gamma = self.gamma
        discounted = np.empty_like(rewards)
        cumulative_sum = 0.
        for i in range(len(rewards) - 1, -1, -1):
            cumulative_sum *= (1 - dones[i]) * gamma
            cumulative_sum += rewards[i]
            discounted[i] = cumulative_sum
        return discounted

    def compute_gae(self, rewards, values, values_next, dones):
        delta = rewards + self.gamma * values_next * (1 - dones) - values
        advantages = self.discount(delta, dones, gamma=self.gamma * self.lmbda, normalize=False)
        returns = advantages + values
        return advantages, returns
