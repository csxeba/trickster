from . import abstract
from . import evaluation
from . import logging
from . import optimization
from . import tensorboard


def get_defaults(history_keys: list = None,
                 testing_rollout=None,
                 log_tensorboard: bool = False,
                 experiment_name: str = None):

    callbacks = []

    if testing_rollout is not None:
        callbacks.append(evaluation.TrajectoryEvaluator(testing_rollout))
    if history_keys is not None:
        callbacks.append(logging.ProgressPrinter(history_keys))
    if log_tensorboard:
        callbacks.append(tensorboard.TensorBoard(logdir="default", experiment_name=experiment_name))

    print(f" [Trickster.callbacks] - Added default callbacks: {', '.join(c.__class__.__name__ for c in callbacks)}")
    return callbacks
