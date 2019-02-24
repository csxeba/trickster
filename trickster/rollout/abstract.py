from typing import List

from ..abstract import AgentBase


class RolloutConfig:

    def __init__(self,
                 max_steps=None,
                 skipframes=None,
                 initial_reward=None):

        self.max_steps = max_steps
        self.skipframes = skipframes or 1
        self.initial_reward = initial_reward or 0.


class RolloutBase:

    def __init__(self, agent: AgentBase, env, config: RolloutConfig=None):
        self.agent = agent
        self.env = env
        self.cfg = config or RolloutConfig()

    def reset_memory(self):
        self.agent.memory.reset()


class MultiRolloutBase:

    def __init__(self,
                 agents: List[AgentBase],
                 envs: list,
                 rollout_configs=None):

        self.num_rollouts = len(envs)
        if self.num_rollouts <= 1:
            raise ValueError("At least 2 environments are required for a MultiRollout!")
        if self.num_rollouts != len(agents):
            raise ValueError("There should be an equal number of agents and envs!")

        agent_ids = {id(agent) for agent in agents}
        if len(agent_ids) < len(agents):
            print("[MultiRollout] - Warning: one or more agents are shared between rollouts!")

        env_ids = {id(env) for env in envs}
        if len(env_ids) < len(envs):
            print("[MultiRollout] - Warning: one or more environments are shared between rollouts!")

        if rollout_configs is None:
            rollout_configs = [RolloutConfig() for _ in range(self.num_rollouts)]
        if isinstance(rollout_configs, RolloutConfig):
            rollout_configs = [rollout_configs] * self.num_rollouts

        self.rollout_configs = rollout_configs
        self.rollouts = None  # type: List[RolloutBase]

    def reset_memory(self):
        for rollout in self.rollouts:
            rollout.reset_memory()

    def gather_memory(self):
        excluded_indices = []
