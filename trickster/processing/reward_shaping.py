import dataclasses
from typing import Union

import numpy as np

from ..utility import numeric_utils


@dataclasses.dataclass
class ShapedReward:

    target_values: np.ndarray = None
    advantages: np.ndarray = None


class ValueTarget:

    BELLMAN = "bellman"
    DISCOUNTED = "discounted"
    GAE_RETURN = "gae_return"


class RewardShaper:

    def __init__(self,
                 discount_factor_gamma: float = 0.99,
                 gae_lambda: Union[float, None] = 0.97,
                 normalize_advantages: bool = True,
                 value_target: str = ValueTarget.DISCOUNTED):

        self.gamma = discount_factor_gamma
        self.lmbda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.value_target_mode = value_target
        print(" [Trickster] - Reward shaping strategy:")
        if gae_lambda:
            print(" [Trickster] - Utilizing Generalized Advantage Estimation")
        if value_target == ValueTarget.BELLMAN:
            print(" [Trickster] - Value target set to Bellman Target")
        elif value_target == ValueTarget.DISCOUNTED:
            print(" [Trickster] - Value target is discounted reward")
        elif value_target == ValueTarget.GAE_RETURN:
            if not gae_lambda:
                raise RuntimeError("GAE returns can only be calcuated is gae_lambda is set")
            print(" [Trickster] - Value target is the GAE return")
        if normalize_advantages:
            print(" [Trickster] - Advantages will be normalized")

    def discount(self, rewards, dones, gamma=None):
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
        advantages = self.discount(delta, dones, gamma=self.gamma * self.lmbda)
        return advantages

    def shape_rewards(self, rewards, dones, values=None, values_next=None) -> ShapedReward:

        returns = self.discount(rewards, dones)

        if self.lmbda:
            advantages = self.compute_gae(rewards, values, values_next, dones)
            if self.value_target_mode == ValueTarget.GAE_RETURN:
                return ShapedReward(advantages + values, advantages)
        else:
            advantages = returns.copy()

        if self.value_target_mode == ValueTarget.BELLMAN:

            target_value = rewards + values_next * self.gamma * dones
        elif self.value_target_mode == ValueTarget.DISCOUNTED:
            target_value = returns
        else:
            assert False

        if values is not None:
            advantages = returns - values

        if self.normalize_advantages:
            advantages = numeric_utils.safe_normalize(advantages)

        return ShapedReward(target_value, advantages)
