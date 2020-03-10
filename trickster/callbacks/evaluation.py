import numpy as np

from .abstract import Callback
from ..rollout import Trajectory
from ..utility.history import History

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
        history.append(**{"RWD/sum": np.sum(rewards), "RWD/std": np.std(rewards)})


class TrajectoryRenderer(Callback):

    def __init__(self,
                 testing_rollout: Trajectory,
                 frequency: int = 100,
                 verbose: int = 1,
                 repeats: int = 5):

        super().__init__()
        self.testing_rollout = testing_rollout
        self.verbose = verbose
        self.repeats = repeats
        self.frequency = frequency

    def on_epoch_end(self, epoch: int, history: History = None):
        if epoch % self.frequency == 0:
            print()
            self.testing_rollout.render(repeats=self.repeats, verbose=self.verbose)
