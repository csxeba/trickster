from . import abstract
from . import evaluation
from . import logging
from . import optimization


def get_defaults(testing_rollout, smoothing_window_size: int = 10):
    cbs = [
        evaluation.TrajectoryEvaluator(testing_rollout),
        evaluation.TrajectoryRenderer(testing_rollout),
        logging.ProgressPrinter(testing_rollout.agent.history_keys),
        logging.HistoryPlotter(smoothing_window_size)
    ]
    return cbs
