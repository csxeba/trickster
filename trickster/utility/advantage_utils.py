import numpy as np


def discount(rewards, dones, gamma):
    discounted = np.empty_like(rewards)
    cumulative_sum = 0.
    for i in range(len(rewards)-1, -1, -1):
        cumulative_sum *= (1 - dones[i]) * gamma
        cumulative_sum += rewards[i]
        discounted[i] = cumulative_sum
    return discounted


def compute_gae(rewards, values, values_next, dones, gamma, lmbda):
    delta = rewards + gamma * values_next * (1 - dones) - values
    advantages = discount(delta, dones, gamma * lmbda)
    return advantages
