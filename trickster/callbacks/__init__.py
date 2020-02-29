from . import abstract
from . import evaluation
from . import logging
from . import optimization
from . import tensorboard


def get_defaults(testing_rollout, log_tensorboard: bool = False):
    cbs = [
        evaluation.TrajectoryEvaluator(testing_rollout),
        evaluation.TrajectoryRenderer(testing_rollout),
        logging.ProgressPrinter(testing_rollout.agent.history_keys),
    ]
    if log_tensorboard:
        cbs.append(tensorboard.TensorBoard(logdir="default"))
    print(" [Trickster.callbacks] - Added default callbacks:")
    for c in cbs:
        print(" [Trickster.callbacks] -", c.__class__.__name__)
    return cbs
