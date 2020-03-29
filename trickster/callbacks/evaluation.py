import pathlib
from typing import Union

import numpy as np

from .abstract import Callback
from ..rollout import Trajectory
from ..utility.history import History
from ..utility.artifactory import Artifactory
from ..utility import render_utils

__all__ = ["TrajectoryEvaluator", "TrajectoryRenderer"]


class TrajectoryEvaluator(Callback):

    def __init__(self, testing_rollout: Trajectory, repeats=4):
        super().__init__()
        self.testing_rollout = testing_rollout
        self.repeats = repeats

    def on_epoch_end(self, epoch: int, history: History = None):
        rewards = []
        for r in range(self.repeats):
            test_history = self.testing_rollout.rollout(verbose=0, push_experience=False)
            rewards.append(test_history["reward_sum"])
        history.append(**{"RWD/sum": np.mean(rewards), "RWD/std": np.std(rewards)})


class TrajectoryRenderer(Callback):

    def __init__(self,
                 testing_rollout: Trajectory,
                 frequency: int = 100,
                 verbose: int = 1,
                 repeats: int = 5,
                 output_to_screen: bool = True,
                 fps: int = 25,
                 scaling_factor: float = 1.,
                 artifactory: Artifactory = "default"):

        super().__init__()
        self.testing_rollout = testing_rollout
        self.verbose = verbose
        self.repeats = repeats
        self.frequency = frequency
        self.output_to_screen = output_to_screen
        self.fps = fps
        self.scaling_factor = scaling_factor
        if artifactory == "default":
            artifactory = Artifactory.make_default()
        self.output_files_directoy = pathlib.Path(artifactory.renders)

    def _make_renderer(self, epoch: int):
        output_file_path = str(self.output_files_directoy / f"render_epoch{epoch:0>7}.avi")
        return render_utils.factory(screen_name=self.testing_rollout.experiment_name,
                                    output_file_path=output_file_path,
                                    fps=self.fps,
                                    scaling_factor=self.scaling_factor)

    def on_epoch_end(self, epoch: int, history: History = None):
        if epoch % self.frequency == 0:
            if self.verbose:
                print()
            renderer = self._make_renderer(epoch)
            self.testing_rollout.render(self.repeats, self.verbose, renderer)
