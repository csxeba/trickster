import numpy as np

from .rollout import Rollout, RolloutConfig


class MultiRollout:

    def __init__(self, agent, envs: list, rollout_configs=None, learning=True, warmup_episodes=0):
        self.num_rollouts = len(envs)
        self.agent = agent
        self.learning = learning
        self.warmup = warmup_episodes

        if rollout_configs is None:
            rollout_configs = [RolloutConfig() for _ in range(self.num_rollouts)]
        if isinstance(rollout_configs, RolloutConfig):
            rollout_configs = [rollout_configs] * self.num_rollouts
        self.rollouts = [Rollout(self.agent, env, config) for env, config in zip(envs, rollout_configs)]

        self.episodes = 0
        self.steps = 0

    @property
    def finished(self):
        return all(rollout.finished for rollout in self.rollouts)

    def roll(self, steps, verbose=1, learning_batch_size=32):
        assert not self.finished
        rewards = np.zeros(self.num_rollouts)
        history = {}

        for i, rollout in enumerate(self.rollouts):
            if rollout.finished:
                continue
            rollout_history = rollout.roll(steps=steps, verbose=verbose, learning_batch_size=0)
            rewards[i] += rollout_history["reward_sum"]

        assert self.agent.memory.N
        self.steps += steps

        if self.episodes >= self.warmup and self.learning and learning_batch_size:
            assert self.agent.memory.N
            result = self.agent.fit(batch_size=learning_batch_size, verbose=verbose)
            for key in result:
                history[key] = result[key]

        history["reward_sum"] = np.mean(rewards)
        history["step"] = max(rollout.step for rollout in self.rollouts)
        history["episode"] = self.episodes
        return history

    def rollout(self, verbose=1, learning_batch_size=0):
        reward_sum = 0.

        for i, rollout in enumerate(self.rollouts):
            rollout.reset()
            if verbose:
                print(f"Rolling environment #{i}")
            history = rollout.rollout(verbose=0, learning_batch_size=0)
            reward_sum += history["reward_sum"]

        reward_sum /= self.num_rollouts
        history = {"reward_sum": reward_sum}

        if learning_batch_size != 0:
            agent_history = self.agent.fit(learning_batch_size, verbose=verbose)
            history.update(agent_history)

        return history

    def reset(self):
        for rollout in self.rollouts:
            rollout.reset()
        self.steps = 0
        self.episodes += 1
