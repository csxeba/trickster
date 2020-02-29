from typing import Union

from .abstract import MultiRolloutBase
from .rolling import Rolling
from .trajectory import Trajectory
from trickster.agent.abstract import RLAgentBase
from ..utility import training_utils


class MultiRolling(MultiRolloutBase):

    def __init__(self, agent: RLAgentBase, envs: list, max_steps: int = None):
        super().__init__(agent, envs, max_steps)
        self.rollouts = [Rolling(agent, env) for env in envs]

    def roll(self,
             steps: int,
             verbose: int = 0,
             push_experience: bool = True):

        """
        Executes the agent in the environment.

        :param steps:
            How many timesteps to run the simulation for.
        :param verbose:
            Level of verbosity.
        :param push_experience:
            Whether to save the obtained data to the agent's learning memory.
        """

        for i, rolling in enumerate(self.rollouts, start=1):
            if verbose:
                print(f" [MultiRolling] - rolling in rollout #{i}:")
            rolling.roll(steps, verbose=verbose, push_experience=push_experience)

    def fit(self,
            epochs: int,
            updates_per_epoch: int = 32,
            steps_per_update: int = 32,
            update_batch_size: int = -1,
            warmup_buffer: Union[bool, int] = True,
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
        :param warmup_buffer: int
            Whether to run some steps so the learning buffer is not empty. True means run for "update_batch_size" steps.
        :param callbacks: List[Callback]
            A list of callbacks or "default".
        :param testing_rollout: Trajectory
            This should be used to test the agent in. Must only be set if callbacks == "default"
        :param log_tensorboard: bool
            Whether to create a TensorBoard log.
        :return: None
        """

        if warmup_buffer is True:
            self.roll(steps=update_batch_size // self.num_rollouts)
        else:
            self.roll(steps=warmup_buffer // self.num_rollouts)

        training_utils.fit(self, epochs, updates_per_epoch, steps_per_update, update_batch_size,
                           testing_rollout, log_tensorboard, callbacks)

    def summary(self):
        pfx = " [Trickster.MultiRolling] -"
        print(pfx, "Summary:")
        print(pfx, "Num rollouts:", self.num_rollouts)
        print(pfx, "Environment:", self.rollouts[0].env.unwrapped.spec.id)
        print(pfx, "Agent:", self.agent.__class__.__name__)
