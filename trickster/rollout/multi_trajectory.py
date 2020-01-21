import numpy as np

from .trajectory import Trajectory
from .abstract import MultiRolloutBase
from trickster.agent.abstract import RLAgentBase


class MultiTrajectory(MultiRolloutBase):

    def __init__(self, agent: RLAgentBase, envs: list, rollout_configs=None):
        super().__init__(agent, envs, rollout_configs)
        self.rollouts = [Trajectory(agent, env, config) for env, config in
                         zip(envs, self.rollout_configs)]

    def rollout(self, verbose=1, push_experience=True):
        rewards = np.empty(self.num_rollouts)

        for i, rollout in enumerate(self.rollouts):
            history = rollout.rollout(verbose=0, push_experience=push_experience)
            rewards[i] = history["reward_sum"]
            if verbose:
                print(f"Rolled out environment #{i}, got total reward of {rewards[i]:.4f}")

        return {"mean_reward": np.mean(rewards), "rewards": rewards}
