import numpy as np
import gym

from trickster.agent import PPO
from trickster.rollout import Trajectory, RolloutConfig

ENV_NAME = "Pong-v0"
TRAJECTORY_MAX_STEPS = None
EPOCHS = 1000
ROLLOUTS_PER_EPOCH = 16


class Pong(gym.ObservationWrapper):

    def __init__(self):
        super().__init__(env=ENV_NAME)
        self.past = None

    @staticmethod
    def _process_frame(I):
        I = I[30:185]
        I = I[::2, ::2, 0].astype("float32")  # 80 x 80
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I

    def observation(self, I):
        I = self._process_frame(I)
        dif = self.past - I
        self.past = I
        return dif

    def reset(self, **kwargs):
        I = self.env.reset(**kwargs)
        self.past = self._process_frame(I)
        return self.past


env = gym.make(ENV_NAME)

agent = PPO.from_environment(env, actor_updates=50, critic_updates=50)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))

rollout.fit(epochs=EPOCHS, rollouts_per_epoch=ROLLOUTS_PER_EPOCH, render_every=10)
rollout.render(repeats=10)
