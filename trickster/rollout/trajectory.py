from typing import Union

import numpy as np

from .abstract import RolloutBase
from ..agent.abstract import RLAgentBase
from ..utility import history, render_utils
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

    def rollout(self,
                verbose=1,
                push_experience=True,
                renderer: render_utils.Renderer = None):

        """
        Execute a complete rollout, until a 'done' flag is received or max_steps is reached.

        :param verbose:
            How much info to print.
        :param push_experience:
            Whether to save the experience for training.
        :param renderer:
            Renderer instance or "default". In the latter case, the rollout will be rendered to screen.
        :param renderer:
            Keyword arguments to be passed to the rendering function.
        """
        self.worker.set_learning_mode(push_experience)
        state = self.env.reset()
        reward = 0.
        done = False
        reward_sum = 0.
        step = 1
        while 1:
            if renderer is not None:
                frame = self.env.render(mode="rgb_array")
                renderer.append(frame)
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

    def render(self,
               repeats: int = 1,
               verbose: int = 1,
               renderer: Union[render_utils.Renderer, str, None] = "default"):

        if renderer in ("default", "screen"):
            renderer = render_utils.ScreenRenderer(self.experiment_name)
            if verbose:
                print(f" [Trickster] - Rendering to screen: {self.experiment_name}")
        elif renderer == "file":
            file = f"{self.experiment_name}_render.mp4"
            renderer = render_utils.FileRenderer(file)
            if verbose:
                print(f" [Trickster] - Rendering to file: {file}")

        for r in range(1, repeats+1):
            if verbose:
                print(f" [Trickster] --- Rendering {r}/{repeats} runs ---")
            self.rollout(verbose, push_experience=False, renderer=renderer)
            if verbose:
                print()

    def fit(self,
            epochs: int,
            updates_per_epoch: int = 1,
            rollouts_per_update: int = 1,
            update_batch_size: int = -1,
            callbacks: list = "default",
            log_tensorboard: bool = False):

        """
        Orchestrates a basic learning scheme.
        :param epochs: int
            The major unit of learning. Determines the frequency of metric logging.
        :param updates_per_epoch: int
            How many updates to consider a learning epoch.
        :param rollouts_per_update: int
            How many rollouts to gather before running an update.
        :param update_batch_size: int
            If set to -1, the complete experience buffer will be used in a single parameter update.
        :param callbacks: List[Callback]
            A list of callbacks or "default".
        :param log_tensorboard: bool
            Whether to log to TensorBoard.
        :return: History
            A History object aggregating the learning metrics.
        """

        train_history = history.History()

        if callbacks is None:
            callbacks = []
        elif callbacks == "default":
            callbacks = _cbs.get_defaults(progress_keys=self.progress_keys,
                                          testing_rollout=None,
                                          log_tensorboard=log_tensorboard,
                                          experiment_name=self.experiment_name)

        callbacks = _cbs.abstract.CallbackList(callbacks)
        callbacks.set_rollout(self)

        callbacks.on_train_begin()

        for epoch in range(1, epochs+1):

            callbacks.on_epoch_begin(epoch, train_history)

            rewards = []

            for update in range(updates_per_epoch):

                callbacks.on_batch_begin()

                for roll in range(rollouts_per_update):
                    rollout_history = self.rollout(verbose=0, push_experience=True)
                    rewards.append(rollout_history["reward_sum"])

                callbacks.on_batch_end()

            agent_history = self.agent.fit(batch_size=update_batch_size)
            agent_history.update({"RWD/sum": np.mean(rewards), "RWD/std": np.std(rewards)})

            train_history.append(**agent_history)

            callbacks.on_epoch_end(epoch, train_history)

        callbacks.on_train_end(train_history)

        return train_history

    def summary(self):
        pfx = " [Trickster.Trajectory] -"
        print(pfx, "Environment:", self.env.unwrapped.spec.id)
        print(pfx, "Agent:", self.agent.__class__.__name__)
