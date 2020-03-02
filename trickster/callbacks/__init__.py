from . import abstract
from . import evaluation
from . import logging
from . import optimization
from . import tensorboard


def get_defaults(testing_rollout, log_tensorboard: bool = False):
    cbs = [
        evaluation.TrajectoryEvaluator(testing_rollout),
        logging.ProgressPrinter(testing_rollout.agent.history_keys),
        evaluation.TrajectoryRenderer(testing_rollout)
    ]
    if log_tensorboard:
        expname = "_".join([testing_rollout.agent.__class__.__name__,
                           testing_rollout.env.spec.id])
        cbs.append(tensorboard.TensorBoard(logdir="default", experiment_name=expname))
    print(" [Trickster.callbacks] - Added default callbacks:")
    for c in cbs:
        print(" [Trickster.callbacks] -", c.__class__.__name__)
    return cbs
