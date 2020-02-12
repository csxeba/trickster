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
        self.worker = agent.create_worker()
        self._rolling_worker = None

    def _sample_action(self):
        return self.worker.sample(self.state, self.reward, self.done)

    def _rolling_job(self):
        while 1:
            self._reset()
            while 1:
                yield self.step
                self.action = self._sample_action()
                if self._finished():
                    break
                assert not self.done

                self.state, self.reward, self.done, self.info = self.env.step(self.action)
                self.step += 1

    def roll(self, steps, verbose=0, learning=True):
        """
        Executes a given number of steps in the environment.
        :param steps: int
            How many steps to execute
        :param verbose: int
            How much info to print
        :param learning: bool
            Whether to save the experience for future learning
        :return: dict
            With keys: "rewards" and "mean_reward"
        """
        if self._rolling_worker is None:
            self._rolling_worker = self._rolling_job()

        rewards = []

        self.worker.set_learning_mode(learning)
        for i, step in enumerate(self._rolling_worker):
            rewards.append(self.reward)
            if verbose:
                print("Step {} rwd: {:.4f}".format(self.step, self.reward))
            if i >= steps:
                break
        self.worker.end_trajectory()
        self.worker.set_learning_mode(False)

        return {"mean_reward": np.mean(rewards), "rewards": np.array(rewards)}

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
            warmup_buffer: bool = False):

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
        :return: None
        """

        if warmup_buffer:
            self.roll(steps=update_batch_size, verbose=0, learning=True)

        training_ops.fit(self, epochs, updates_per_epoch, steps_per_update, update_batch_size,
                         testing_rollout, plot_curves, render_every)
