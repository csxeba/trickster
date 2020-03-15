from typing import Union

from .trajectory import Trajectory
from .abstract import RolloutBase
from ..agent.abstract import RLAgentBase
from ..utility import training_utils


__all__ = ["Rolling"]


class Rolling(RolloutBase):

    """Generate n-step trajectories for eg. Time-Difference learning"""

    def __init__(self, agent: RLAgentBase, env, max_steps: int = None):

        """
        :param agent:
            A reinforcement learning agent.
        :param env:
            An environment supporting the Gym interface.
        :param max_steps:
            Max number of steps before termination of an episode.
        """

        super().__init__(agent, env, max_steps)

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
        return self.worker.sample(self.state.astype("float32"), self.reward, self.done)

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
                if self._finished(self.done, self.step):
                    break
                assert not self.done

                self.state, self.reward, self.done, self.info = self.env.step(self.action)
                self.step += 1

    def roll(self, steps, verbose=0, push_experience=True, random_actions=False):
        """
        Executes a given number of steps in the environment.
        :param steps: int
            How many steps to execute.
        :param verbose: int
            How much info to print.
        :param push_experience: bool
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

        self.worker.set_learning_mode(push_experience)
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
        self.reward = 0.
        self.info = {}
        self.done = False
        self.state = self.env.reset()
        self.step = 0
        self.episodes += 1

    def fit(self,
            epochs: int,
            updates_per_epoch: int = 32,
            steps_per_update: int = 32,
            update_batch_size: int = -1,
            buffer_warmup: Union[bool, int] = True,
            callbacks: list = "default",
            testing_rollout: Trajectory = None,
            log_tensorboard: bool = False):

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
        :param buffer_warmup: int
            Whether to run some steps so the learning buffer is not empty. True means run for "update_batch_size" steps.
        :param callbacks: List[Callback]
            A list of callbacks or "default".
        :param testing_rollout: Trajectory
            This should be used to test the agent in. Must only be set if callbacks == "default"
        :param log_tensorboard: bool
            Whether to create a TensorBoard log.
        """

        if buffer_warmup is True and update_batch_size > 0:
            self.roll(steps=update_batch_size, verbose=0, push_experience=True, random_actions=False)
        elif buffer_warmup > 0:
            self.roll(steps=buffer_warmup, verbose=0, push_experience=True, random_actions=False)

        training_utils.fit(self, epochs, updates_per_epoch, steps_per_update, update_batch_size,
                           testing_rollout, log_tensorboard, callbacks)
