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


class MultiRolloutBase:

    def __init__(self, agent, envs: list, rollout_configs=None):
        self.num_rollouts = len(envs)
        if self.num_rollouts <= 1:
            raise ValueError("At least 2 environments are required for a MultiRollout!")
        self.agent = agent

        env0 = envs[0]
        for env in envs[1:]:
            if env is env0:
                print(" [MultiRollout] - Warning: one or more environments are shared between rollouts!")

        if rollout_configs is None:
            rollout_configs = [RolloutConfig() for _ in range(self.num_rollouts)]
        if isinstance(rollout_configs, RolloutConfig):
            rollout_configs = [rollout_configs] * self.num_rollouts

        self.rollout_configs = rollout_configs
