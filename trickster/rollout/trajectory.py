from .abstract import RolloutBase
from ..agent.abstract import RLAgentBase
from ..utility import history
from .. import callbacks as _cbs


class Trajectory(RolloutBase):

    """Generate complete trajectories for Monte Carlo learning or testing purposes"""

    def __init__(self, agent: RLAgentBase, env, max_steps: int = None):

        """
        :param agent:
            A reinforcement learning agent.
        :param env:
            An environment supporting the Gym interface.
        :param max_steps:
            Max number of steps before termination of an episode.
        """

        super().__init__(agent, env, max_steps)
        self.episodes = 0
        self.worker = agent.create_worker()

    def rollout(self, verbose=1, push_experience=True, render=False):
        """
        Execute a complete rollout, until a 'done' flag is received or max_steps is reached.

        :param verbose:
            How much info to print.
        :param push_experience:
            Whether to save the experience for training.
        :param render:
            Whether to display the rollout visually.
        """

        self.worker.set_learning_mode(push_experience)
        state = self.env.reset()
        reward = 0.
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
                print(f"\rStep: {step} total reward: {reward_sum:.4f} action: {action}", end="")
            step += 1
        if verbose:
            print()
        self.worker.end_trajectory()
        self.worker.set_learning_mode(False)

        return {"reward_sum": reward_sum, "steps": step}

    def fit(self,
            epochs: int,
            rollouts_per_epoch: int = 1,
            update_batch_size: int = -1,
            callbacks: list = "default",
            log_tensorboard: bool = False):

        """
        Orchestrates a basic learning scheme.
        :param epochs: int
            The major unit of learning. Determines the frequency of metric logging.
        :param rollouts_per_epoch: int
            How many rollouts (or episodes) an epoch consits of.
        :param update_batch_size: int
            If set to -1, the complete experience buffer will be used in a single parameter update.
        :param callbacks: List[Callback]
            A list of callbacks or "default".
        :param log_tensorboard: bool
            Whether to log to TensorBoard
        :return: History
            A History object aggregating the learning metrics
        """

        train_history = history.History("reward_sum", *self.agent.history_keys)

        if callbacks is None:
            callbacks = []
        if callbacks == "default":
            cbs = [
                _cbs.logging.ProgressPrinter(self.agent.history_keys),
                _cbs.evaluation.TrajectoryRenderer(self)
            ]
            if log_tensorboard:
                expname = "_".join([self.agent.__class__.__name__,
                                    self.env.spec.id])
                cbs.append(_cbs.tensorboard.TensorBoard(logdir="default", experiment_name=expname))
            print(" [Trickster.Trajectory] - Added default callbacks:")
            for c in cbs:
                print(" [Trickster.Trajectory] -", c.__class__.__name__)

        callbacks = _cbs.abstract.CallbackList(callbacks)
        callbacks.on_train_begin()

        for epoch in range(1, epochs+1):

            callbacks.on_epoch_begin(epoch, train_history)

            for roll in range(rollouts_per_epoch):
                callbacks.on_batch_begin()
                rollout_history = self.rollout(verbose=0, push_experience=True)
                train_history.buffer(reward_sum=rollout_history["reward_sum"])
                callbacks.on_batch_end()

            agent_history = self.agent.fit(batch_size=update_batch_size)

            train_history.push_buffer()
            train_history.append(**agent_history)

            callbacks.on_epoch_end(epoch, train_history)

        callbacks.on_train_end(train_history)

        return train_history

    def render(self, repeats=1, verbose=1):
        for r in range(1, repeats+1):
            if verbose:
                print(f" --- Rendering {r}/{repeats} runs ---")
            self.rollout(verbose, push_experience=False, render=True)
            if verbose:
                print()

    def summary(self):
        pfx = " [Trickster.Trajectory] -"
        print(pfx, "Environment:", self.env.unwrapped.spec.id)
        print(pfx, "Agent:", self.agent.__class__.__name__)
