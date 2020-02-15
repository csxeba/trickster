from trickster.agent.abstract import RLAgentBase
from .abstract import RolloutBase, RolloutConfig
from ..utility import history, visual, progress_utils


class Trajectory(RolloutBase):

    """Generate complete trajectories for Monte Carlo learning or testing purposes"""

    def __init__(self, agent: RLAgentBase, env, config: RolloutConfig=None):
        super().__init__(agent, env, config)
        self.episodes = 0
        self.worker = agent.create_worker()

    def rollout(self, verbose=1, push_experience=True, render=False):
        """
        Execute a complete rollout, eg. until a 'done' flag is received

        :param verbose: how much info to print
        :param push_experience: whether to save the experience for training
        :param render: whether to display the rollout visually
        :return: dict
            With keys: "reward_sum" and "steps".
        """

        self.worker.set_learning_mode(push_experience)
        state = self.env.reset()
        reward = self.cfg.initial_reward
        done = False
        reward_sum = 0.
        step = 1
        while 1:
            if render:
                self.env.render()
            action = self.worker.sample(state.astype("float32"), reward, done)
            if self._finished(done, step):
                break
            state, reward, done, info = self.env.step(action)
            if not isinstance(reward, float):
                reward = float(reward)
            reward_sum += reward
            if verbose:
                print("\rStep: {} total reward: {:.4f}".format(step, reward_sum), end="")
            step += 1
        if verbose:
            print()
        if not done:
            self.worker.end_trajectory()
        self.worker.set_learning_mode(False)

        return {"reward_sum": reward_sum, "steps": step}

    def _finished(self, current_done_value, current_step):
        done = current_done_value
        if self.cfg.max_steps is not None:
            done = done or current_step >= self.cfg.max_steps
        return done

    def fit(self, epochs, rollouts_per_epoch=1, update_batch_size=-1, smoothing_window_size=10, plot_curves=True,
            render_every=100):

        """
        Orchestrates a basic learning scheme.
        :param epochs: int
            The major unit of learning. Determines the frequency of metric logging.
        :param rollouts_per_epoch: int
            How many rollouts (or episodes) an epoch consits of.
        :param update_batch_size: int
            If set to -1, the complete experience buffer will be used in a single parameter update.
        :param smoothing_window_size: int
            Metric smoothing for console output readability. Defined in epochs.
        :param plot_curves: bool
            Whether to plot the agent's metrics. smoothing_window_size will also be applied.
        :param render_every: int
            Frequency of rendering a trajectory. Defined in epochs.
        :return: History
            A History object aggregating the learning metrics
        """

        train_history = history.History("reward_sum", *self.agent.history_keys)

        print()
        progress_logger = progress_utils.ProgressPrinter(keys=train_history.keys)
        progress_logger.print_header()

        for epoch in range(1, epochs+1):
            for roll in range(rollouts_per_epoch):
                rollout_history = self.rollout(verbose=0, push_experience=True)
                train_history.buffer(reward_sum=rollout_history["reward_sum"])

            agent_history = self.agent.fit(batch_size=update_batch_size)

            train_history.push_buffer()
            train_history.append(**agent_history)
            progress_logger.print(train_history, average_last=smoothing_window_size, return_carriege=True)

            if epoch % smoothing_window_size == 0:
                print()

            if epoch % (smoothing_window_size*10) == 0:
                print()
                progress_logger.print_header()

            if render_every > 0:
                if epoch % render_every == 0:
                    self.render(repeats=3)

        if plot_curves:
            visual.plot_history(train_history, smoothing_window_size, skip_first=0, show=True)

        return train_history

    def render(self, repeats=1, verbose=1):
        for r in range(repeats):
            if verbose:
                print(f" --- Rendeding {r}/{repeats} runs ---")
            self.rollout(verbose, push_experience=False, render=True)
            if verbose:
                print()
