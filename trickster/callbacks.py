import numpy as np
from tensorflow import keras

from trickster.rollout import Trajectory
from trickster.abstract import RLAgentBase


class TricksterCallback:

    def __init__(self):
        self.agent = None  # type: RLAgentBase

    def set_agent(self, agent: RLAgentBase):
        self.agent = agent

    def on_update_begin(self, update, logs=None):
        pass

    def on_update_end(self, update, logs=None):
        pass

    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        pass

    def on_train_begin(self, episode, logs=None):
        pass

    def on_train_end(self, episode, logs=None):
        pass


class KerasWrapper(TricksterCallback):

    def __init__(self, keras_callback: keras.callbacks.Callback):
        super().__init__()
        self.on_update_begin = keras_callback.on_batch_begin
        self.on_update_end = keras_callback.on_batch_end
        self.on_episode_begin = keras_callback.on_epoch_begin
        self.on_episode_end = keras_callback.on_epoch_end
        self.on_train_begin = keras_callback.on_train_begin
        self.on_train_end = keras_callback.on_train_end


class EvaluationCallback(TricksterCallback):

    def __init__(self, env, rollout_config, repeats=1):
        super().__init__()
        self.env = env
        self.cfg = rollout_config
        self.repeats = repeats
        self.trajectory = None  # type: Trajectory

    def set_agent(self, agent):
        super().set_agent(agent)
        self.trajectory = Trajectory(self.agent, self.env, self.cfg)

    def on_episode_end(self, episode, logs=None):
        rewards = []
        for repeat in range(self.repeats):
            history = self.trajectory.rollout(verbose=0, push_experience=False, render=False)
            rewards.append(history["reward_sum"])
        logs = logs or {}
        logs["rewards"] = np.mean(rewards)


class RewardCheckpoint(TricksterCallback):

    def __init__(self,
                 output_root,
                 monitor="reward_sum",
                 mode="max",
                 save_best_only=False,
                 overwrite=False):

        super().__init__()
        self.output_root = output_root
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.overwrite = overwrite

    def on_episode_end(self, episode, logs=None):
        logs = logs or {}

