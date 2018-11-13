import numpy as np


def discount_reward(R, gamma):
    discounted = np.zeros_like(R)
    cumulative_sum = 0.
    for i, r in enumerate(R[::-1]):
        cumulative_sum *= gamma
        cumulative_sum += r
        discounted[i] = cumulative_sum
    return discounted[::-1]
