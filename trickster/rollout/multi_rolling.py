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

    def fit(self, episodes, updates_per_episode=32, steps_per_update=32, update_batch_size=-1,
            testing_rollout: Trajectory=None, plot_curves=True, render_every=0):

            training_ops.fit(self,
                             episodes=episodes,
                             updates_per_episode=updates_per_episode,
                             steps_per_update=steps_per_update,
                             update_batch_size=update_batch_size,
                             testing_rollout=testing_rollout,
                             plot_curves=plot_curves,
                             render_every=render_every)
