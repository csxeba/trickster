from typing import List

from ..abstract import RLAgentBase


class RolloutConfig:

    def __init__(self,
                 max_steps=None,
                 skipframes=None,
                 initial_reward=None,
                 testing_rollout=False):

        self.max_steps = max_steps
        self.skipframes = skipframes or 1
        self.initial_reward = initial_reward or 0.
        self.testing_rollout = testing_rollout


class RolloutBase:

    def __init__(self, agent: RLAgentBase, env, config: RolloutConfig=None):
        self.agent = agent
        self.env = env
        self.cfg = config or RolloutConfig()

    def reset_memory(self):
        self.agent.memory.reset()


class MultiRolloutBase:

    def __init__(self,
                 agent: RLAgentBase,
                 envs: list,
                 rollout_configs=None):

        self.agent = agent
        self.num_rollouts = len(envs)
        if self.num_rollouts <= 1:
            raise ValueError("At least 2 environments are required for a MultiRollout!")
        if rollout_configs is None:
            rollout_configs = [RolloutConfig() for _ in range(self.num_rollouts)]
        if isinstance(rollout_configs, RolloutConfig):
            rollout_configs = [rollout_configs] * self.num_rollouts

        self.rollout_configs = rollout_configs
        self.rollouts = None  # type: List[RolloutBase]

    def reset_memory(self):
        for rollout in self.rollouts:
            rollout.reset_memory()
