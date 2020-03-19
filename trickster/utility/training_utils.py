import tensorflow as tf

from . import history
from .. import callbacks as _cbs
from ..rollout import Trajectory


def fit(rolling,
        epochs: int,
        updates_per_epoch: int = 32,
        steps_per_update: int = 32,
        update_batch_size: int = -1,
        testing_rollout: Trajectory = None,
        log_tensorboard: bool = False,
        callbacks: list = "default"):

    if callbacks == "default":
        callbacks = _cbs.get_defaults(rolling.progress_keys,
                                      testing_rollout,
                                      log_tensorboard,
                                      rolling.experiment_name)
    if callbacks is None:
        callbacks = []

    callbacks = _cbs.abstract.CallbackList(list(callbacks))
    callbacks.set_rollout(rolling)

    logger = history.History()

    callbacks.on_train_begin()

    for epoch in range(1, epochs + 1):

        callbacks.on_epoch_begin(epoch)

        rollout_histories = []

        for update in range(1, updates_per_epoch + 1):

            rollout_history = rolling.roll(steps=steps_per_update, verbose=0, push_experience=True)
            agent_history = rolling.agent.fit(batch_size=update_batch_size)
            logger.buffer(**agent_history)
            rollout_histories.append(rollout_history)

        logger.push_buffer()
        logger.append(**rollout_histories[-1])

        callbacks.on_epoch_end(epoch, logger)

    callbacks.on_train_end(logger)

    return logger
