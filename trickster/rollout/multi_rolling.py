from typing import Union

import numpy as np

from .abstract import MultiRolloutBase
from .rolling import Rolling
from .trajectory import Trajectory
from trickster.agent.abstract import RLAgentBase
from ..utility import training_ops


class MultiRolling(MultiRolloutBase):

    def __init__(self, agent: RLAgentBase, envs: list, rollout_configs=None):
        super().__init__(agent, envs, rollout_configs)
        self.rollouts = [Rolling(agent, env, config) for env, config in
                         zip(envs, self.rollout_configs)]

    def roll(self,
             steps: int,
             verbose: int = 0,
             learning: bool = True):

        for i, rolling in enumerate(self.rollouts, start=1):
            if verbose:
                print(f"Rolling in rollout #{i}:")
            rolling.roll(steps, verbose=verbose, learning=learning)

    def fit(self,
            epochs: int,
            updates_per_epoch: int = 32,
            steps_per_update: int = 32,
            update_batch_size: int = -1,
            testing_rollout: Trajectory = None,
            plot_curves: bool = True,
            render_every: int = 0,
            warmup_buffer: Union[bool, int] = False):

        if warmup_buffer is True:
            self.roll(steps=update_batch_size // self.num_rollouts)
        else:
            self.roll(steps=warmup_buffer // self.num_rollouts)

        training_ops.fit(self,
                         episodes=epochs,
                         updates_per_episode=updates_per_epoch,
                         steps_per_update=steps_per_update,
                         update_batch_size=update_batch_size,
                         testing_rollout=testing_rollout,
                         plot_curves=plot_curves,
                         render_every=render_every)
