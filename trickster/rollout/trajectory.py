from ..abstract import RLAgentBase
from .abstract import RolloutBase, RolloutConfig
from ..utility import history, visual


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
        step = 0
        while not self._finished(done, step):
            step += 1
            if render:
                self.env.render()
            action = self.worker.sample(state, reward, done)
            state, reward, done, info = self.env.step(action)
            reward_sum += reward
            if verbose:
                print("\rStep: {} total reward: {:.4f}".format(step, reward_sum), end="")
        if verbose:
            print()
        if push_experience:
            self.worker.push_experience(state, reward, done)
        self.worker.set_learning_mode(False)

        return {"reward_sum": reward_sum, "steps": step}

    def _finished(self, current_done_value, current_step):
        done = current_done_value
        if self.cfg.max_steps is not None:
            done = done or current_step >= self.cfg.max_steps
        return done

    def fit(self, episodes, rollouts_per_update=1, update_batch_size=-1, smoothing_window_size=10, plot_curves=True):

        """
        Orchestrates a basic learning scheme.
        :param episodes: int
            How many episodes to learn for
        :param rollouts_per_update: int
            How many updates an episode consits of
        :param update_batch_size: int
            If set to -1, the complete experience buffer will be used as a single batch
        :param smoothing_window_size: int
            Used together with the argument <plot_curves>
        :param plot_curves: bool
            Whether to plot the agent's metrics
        :return: History
            A History object aggregating the learning metrics
        """

        logger = history.History("reward_sum", *self.agent.history_keys)
        logger.print_header()

        for episode in range(1, episodes+1):
            for update in range(rollouts_per_update):
                rollout_history = self.rollout(verbose=0, push_experience=True)
                logger.buffer(reward_sum=rollout_history["reward_sum"])

            agent_history = self.agent.fit(batch_size=update_batch_size)

            logger.push_buffer()
            logger.record(**agent_history)
            logger.print(average_last=smoothing_window_size, return_carriege=True)

            if episode % smoothing_window_size == 0:
                print()

            if episode % (smoothing_window_size*10) == 0:
                print()
                logger.print_header()

        if plot_curves:
            visual.plot_history(logger, smoothing_window_size, skip_first=0, show=True)

        return logger

    def render(self, repeats=1, verbose=1):
        for r in range(repeats):
            self.rollout(verbose, push_experience=False, render=True)
            print()
