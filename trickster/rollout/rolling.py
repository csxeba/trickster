import numpy as np

from .trajectory import Trajectory
from .abstract import RolloutBase, RolloutConfig
from ..abstract import RLAgentBase
from ..utility import spaces, training_ops


class Rolling(RolloutBase):

    """Generate n-step trajectories for Time-Difference learning"""

    def __init__(self, agent: RLAgentBase, env, config: RolloutConfig=None):

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
                if self.step % self.cfg.skipframes == 0:
                    self.action = self._sample_action()
                if self._finished():
                    break
                assert not self.done

                if isinstance(self.agent.action_space, np.ndarray):
                    a = self.agent.action_space[self.action]
                elif self.agent.action_space == spaces.CONTINUOUS:
                    a = self.action
                else:
                    assert False

                self.state, self.reward, self.done, self.info = self.env.step(a)
                self.step += 1

    def roll(self, steps, verbose=0, push_experience=True):
        """
        Executes a given number of steps in the environment.
        :param steps: int
            How many steps to execute
        :param verbose: int
            How much info to print
        :param push_experience: bool
            Whether to save the experience for future learning
        :return: dict
            With keys: "rewards" and "mean_reward"
        """
        if self._rolling_worker is None:
            self._rolling_worker = self._rolling_job()

        rewards = []

        self.worker.set_learning_mode(push_experience)
        for i, step in enumerate(self._rolling_worker):
            rewards.append(self.reward)
            if verbose:
                print("Step {} rwd: {:.4f}".format(self.step, self.reward))
            if i >= steps:
                break

        if push_experience:
            self.worker.push_experience(self.state, self.reward, self.done)

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

    def fit(self, episodes, updates_per_episode=32, step_per_update=32, update_batch_size=-1,
            testing_rollout: Trajectory=None, plot_curves=True, render_every=0):

        """
        Orchestrates a basic learning scheme.
        :param episodes: int
            How many episodes to learn for
        :param updates_per_episode: int
            How many updates an episode consits of
        :param step_per_update: int
            How many steps to run the agent for in the environment before updating
        :param update_batch_size: int
            If set to -1, the complete experience buffer will be used as a single batch
        :param testing_rollout: Trajectory
            This should be used to test the agent in
        :param plot_curves:
            Whether to plot the agent's metrics
        :return: None
        """

        training_ops.fit(self, episodes, updates_per_episode, step_per_update, update_batch_size,
                         testing_rollout, plot_curves, render_every)
