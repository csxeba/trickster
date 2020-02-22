from typing import Union

import numpy as np

from .trajectory import Trajectory
from .abstract import RolloutBase, RolloutConfig
from ..agent.abstract import RLAgentBase
from ..utility import training_ops


__all__ = ["Rolling"]


class Rolling(RolloutBase):

    """Generate n-step trajectories for Time-Difference learning"""

    def __init__(self, agent: RLAgentBase, env, config: RolloutConfig = None):

        """
        :param agent: A reinforcement learning agent
        :param env: An environment supporting the Gym interface
        :param config: configurations
        """

        super().__init__(agent, env, config)

        self.step = 0
        self.episodes = 0
        self.state = None
        self.action = None
        self.reward = None
        self.info = None
        self.done = None
        self.random_actions = False
        self.worker = agent.create_worker()
        self._rolling_worker = None

    def _sample_action(self):
        return self.worker.sample(self.state, self.reward, self.done)

    def _random_action(self):
        return self.env.action_space.sample()

    def _rolling_job(self):
        while 1:
            self._reset()
            while 1:
                yield self.step
                if self.random_actions:
                    self.action = self._random_action()
                else:
                    self.action = self._sample_action()
                if self._finished():
                    break
                assert not self.done

                self.state, self.reward, self.done, self.info = self.env.step(self.action)
                self.step += 1

    def roll(self, steps, verbose=0, learning=True, random_actions=False):
        """
        Executes a given number of steps in the environment.
        :param steps: int
            How many steps to execute.
        :param verbose: int
            How much info to print.
        :param learning: bool
            Whether to save the experience for future learning.
        :param random_actions: bool
            Whether to take random actions or use the policy.
        :return: dict
            With keys: "rewards" and "mean_reward"
        """
        if self._rolling_worker is None:
            self._rolling_worker = self._rolling_job()

        self.random_actions = random_actions

        rewards = []

        self.worker.set_learning_mode(learning)
        for i, step in enumerate(self._rolling_worker):
            rewards.append(self.reward)
            if verbose:
                print("\r Rolling - Step {}/{} rwd: {: .4f}".format(i, steps, self.reward), end="")
            if i >= steps:
                break
        if verbose:
            print()
        self.worker.end_trajectory()
        self.worker.set_learning_mode(False)
        self.random_actions = False

    def _reset(self):
        self.reward = self.cfg.initial_reward
        self.info = {}
        self.done = False
        self.state = self.env.reset()
        self.step = 0
        self.episodes += 1

    def _finished(self):
        done = self.done
        if self.cfg.max_steps is not None:
            done = done or self.step >= self.cfg.max_steps
        return done

    def fit(self,
            epochs: int,
            updates_per_epoch: int = 32,
            steps_per_update: int = 1,
            update_batch_size: int = 32,
            testing_rollout: Trajectory = None,
            plot_curves: bool = True,
            render_every: int = 0,
            warmup_buffer: Union[bool, int] = False,
            smoothing_window_size: int = 10):

        """
        Orchestrates a basic learning scheme.

        :param epochs: int
            How many episodes to learn for
        :param updates_per_epoch: int
            How many updates an episode consits of
        :param steps_per_update: int
            How many steps to run the agent for in the environment before updating
        :param update_batch_size: int
            If set to -1, the complete experience buffer will be used as a single batch
        :param testing_rollout: Trajectory
            This should be used to test the agent in
        :param plot_curves: bool
            Whether to plot the agent's metrics
        :param render_every: int
            Frequency of rendering, measured in epochs.
        :param warmup_buffer: int
            Whether to fill the memory buffer with data from full-epsilon.
        :param smoothing_window_size: int
            Size of the window used for mean and std calculations.
        :return: None
        """

        if warmup_buffer is True:
            self.roll(steps=update_batch_size, verbose=0, learning=True, random_actions=False)
        elif warmup_buffer:
            self.roll(steps=warmup_buffer, verbose=0, learning=True, random_actions=False)

        training_ops.fit(self, epochs, updates_per_epoch, steps_per_update, update_batch_size,
                         testing_rollout, plot_curves, render_every, smoothing_window_size)
