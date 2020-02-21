from . import history, visual, progress_utils


def fit(rolling, episodes, updates_per_episode=32, steps_per_update=32, update_batch_size=-1,
        testing_rollout=None, plot_curves=True, render_every=0, smoothing_window_size: int = 10):

    logger = history.History("reward_sum", *rolling.agent.history_keys)
    progress_tracker = progress_utils.ProgressPrinter(logger.keys)
    progress_tracker.print_header()

    for episode in range(1, episodes + 1):

        for update in range(1, updates_per_episode+1):

            rolling.roll(steps=steps_per_update, verbose=0, learning=True)
            agent_history = rolling.agent.fit(batch_size=update_batch_size)
            logger.buffer(**agent_history)

        logger.push_buffer()

        if testing_rollout is not None:
            test_history = testing_rollout.rollout(verbose=0, push_experience=False)
            logger.append(reward_sum=test_history["reward_sum"])

        progress_tracker.print(logger, average_last=smoothing_window_size, return_carriege=True)

        if episode % smoothing_window_size == 0:
            print()

        if render_every and testing_rollout is not None and episode % render_every == 0:
            testing_rollout.render(repeats=5)

        if episode % (smoothing_window_size*10) == 0:
            print()
            progress_tracker.print_header()

    if plot_curves:
        visual.plot_history(logger, smoothing_window_size=10)
