from . import history, visual


def fit(rolling, episodes, updates_per_episode=32, step_per_update=32, update_batch_size=-1, testing_rollout=None, plot_curves=True):

    episode_w = len(str(episodes))
    logger = history.History("reward_sum", *rolling.agent.history_keys)

    for episode in range(1, episodes + 1):

        for update in range(updates_per_episode):
            roll_history = rolling.roll(steps=step_per_update, verbose=0, push_experience=True)
            agent_history = rolling.agent.fit(batch_size=update_batch_size)
            logger.buffer(**agent_history)
            if testing_rollout is None:
                logger.buffer(reward_sum=roll_history["reward_sum"])

        logger.push_buffer()

        if testing_rollout is not None:
            test_history = testing_rollout.rollout(verbose=0, push_experience=False)
            logger.record(reward_sum=test_history["reward_sum"])

        logger.print(average_last=10, return_carriege=True, prefix="Episode {:>{w}}".format(episode, w=episode_w))

        if episode % 10 == 0:
            print()

    if plot_curves:
        visual.plot_history(logger, smoothing_window_size=10)
