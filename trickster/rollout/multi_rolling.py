import numpy as np

from .abstract import MultiRolloutBase
from .rolling import Rolling
from .trajectory import Trajectory
from ..abstract import RLAgentBase
from ..utility import training_ops


class MultiRolling(MultiRolloutBase):

    def __init__(self, agent: RLAgentBase, envs: list, rollout_configs=None):
        super().__init__(agent, envs, rollout_configs)
        self.rollouts = [Rolling(agent, env, config) for env, config in
                         zip(envs, self.rollout_configs)]

    def roll(self, steps, verbose=0, push_experience=True):
        rewards = np.empty(self.num_rollouts)

        for i, rolling in enumerate(self.rollouts):
            roll_history = rolling.roll(steps, verbose=0, push_experience=push_experience)
            rewards[i] = roll_history["mean_reward"]
            if verbose:
                print("Rolled in env #{}, got total reward of {:.4f}".format(i, rewards[i]))

        return {"mean_reward": rewards.mean(), "rewards": rewards}

    def fit(self, episodes, updates_per_episode=32, step_per_update=32, testing_rollout: Trajectory=None,
            plot_curves=True):
            training_ops.fit(self, episodes, updates_per_episode, step_per_update, testing_rollout, plot_curves)
