import numpy as np


def discount_reward(R, gamma):
    discounted = np.zeros_like(R)
    cumulative_sum = 0.
    for i, r in enumerate(R[::-1]):
        cumulative_sum *= gamma
        cumulative_sum += r
        discounted[i] = cumulative_sum
    return discounted[::-1]


def compute_gae(R, V, V_next, F, gamma, lmbda):
    delta = R + gamma * V_next * F - V
    advantage_estimation = np.empty_like(delta)
    gae = 0.
    gl = gamma * lmbda
    F = F[::-1]
    for i in reversed(range(len(delta))):
        gae = delta[i] + gl * gae * F[i]
        advantage_estimation[i] = gae
    return advantage_estimation[::-1]


def batch_compute_gae(R, V, V_next, gamma, lmbda):
    delta = R + gamma * V_next - V
    delta[:, -1] = R[:, -1] - V
    advantage_estimation = np.empty_like(delta)
    gae = np.zeros(len(R))
    gl = gamma * lmbda
    for i, d in enumerate(delta[:, ::-1]):
        gae = d + gl * gae
        advantage_estimation[:, i] = gae
    advantage_estimation[:, 0] = delta[:, 0]
    return advantage_estimation[:, ::-1]


def noop(*args):
    if len(args) == 1:
        return args[0]
    return args
