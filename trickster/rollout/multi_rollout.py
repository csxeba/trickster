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
        self.rollouts = [Rollout(env, self.agent, config) for env, config in zip(envs, rollout_configs)]

        self.episodes = 0
        self.steps = 0
        self.rollouts_finished = None

    @property
    def finished(self):
        if self.rollouts_finished is None:
            raise RuntimeError("You should call MultiRollout.reset() first!")
        return all(self.rollouts_finished)

    def roll(self, steps, verbose=1, learning_batch_size=32):
        rewards = np.zeros(self.num_rollouts)
        history = {}

        for i, rollout in enumerate(self.rollouts):
            if rollout.finished:
                if not self.rollouts_finished[i]:
                    self.rollouts_finished[i] = True
                    if verbose:
                        print("Environment {} is finished @ step {}".format(i, rollout.step))
                continue
            reward = rollout.roll(steps=steps, verbose=verbose)
            rewards[i] += reward

        self.steps += steps

        if self.episodes >= self.warmup and self.learning and not self.finished:
            result = self.agent.fit(batch_size=learning_batch_size, verbose=0)
            for key in result:
                history[key] = result[key]

        history["reward_sum"] = np.mean(rewards)
        history["step"] = max(rollout.step for rollout in self.rollouts)
        history["episode"] = self.episodes
        return history

    def reset(self):
        for rollout in self.rollouts:
            rollout.reset()
        self.steps = 0
        self.episodes += 1
        self.rollouts_finished = [False] * self.num_rollouts
