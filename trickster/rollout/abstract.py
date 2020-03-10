from typing import List

from trickster.agent.abstract import RLAgentBase


class RolloutInterface:

    def reset_memory(self):
        raise NotImplementedError

    @property
    def experiment_name(self):
        raise NotImplementedError

    @property
    def history_keys(self):
        raise NotImplementedError


class RolloutBase(RolloutInterface):

    def __init__(self, agent: RLAgentBase, env, max_steps: int = None):
        self.agent = agent
        self.env = env
        self.max_steps = max_steps

    def reset_memory(self):
        self.agent.transition_memory.reset()

    def _finished(self, current_done_value, current_step):
        done = current_done_value
        if self.max_steps is not None:
            done = done or current_step >= self.max_steps
        return done

    @property
    def experiment_name(self):
        return "_".join([self.agent.__class__.__name__, self.env.spec.id])

    @property
    def history_keys(self):
        return ["RWD/sum", "RWD/std"] + self.agent.history_keys


class MultiRolloutBase(RolloutInterface):

    def __init__(self,
                 agent: RLAgentBase,
                 envs: list,
                 max_steps: int = None):

        self.agent = agent
        self.num_rollouts = len(envs)
        if self.num_rollouts <= 1:
            raise ValueError("At least 2 environments are required for a MultiRollout!")
        self.rollouts = None  # type: List[RolloutBase]
        self.max_steps = max_steps

    def reset_memory(self):
        for rollout in self.rollouts:
            rollout.reset_memory()

    @property
    def experiment_name(self):
        return "_".join([self.agent.__class__.__name__, self.rollouts[0].env.spec.id])

    @property
    def history_keys(self):
        return ["RWD/sum", "RWD/std"] + self.agent.history_keys
