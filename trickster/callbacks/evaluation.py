from .abstract import Callback
from ..rollout import Trajectory
from ..utility.history import History

__all__ = ["TrajectoryEvaluator", "TrajectoryRenderer"]


class TrajectoryEvaluator(Callback):

    def __init__(self, testing_rollout: Trajectory):
        super().__init__()
        self.testing_rollout = testing_rollout

    def on_epoch_end(self, epoch: int, history: History = None):
        test_history = self.testing_rollout.rollout(verbose=0, push_experience=False)
        history.append(reward_sum=test_history["reward_sum"])


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
