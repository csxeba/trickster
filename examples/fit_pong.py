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
        self.stack = np.empty([80, 80, 3], dtype="float32")
        self.stack_counter = 0
        self.blank = np.zeros_like(self.stack)

    def push(self, I):
        self.stack[..., -1] = I

    def observation(self, I):

        I = I[30:185]
        I = I[::2, ::2, 0]  # 80 x 80
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1

        self.stack = np.concatenate([self.stack[..., 1:], I[..., None].astype("float32")], axis=-1)
        if self.stack_counter < 3:
            self.stack_counter += 1
            return self.blank

        return self.stack

    def reset(self, **kwargs):
        self.stack_counter = 0
        super().reset(**kwargs)


env = gym.make(ENV_NAME)

agent = PPO.from_environment(env, actor_updates=50, critic_updates=50)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))

rollout.fit(epochs=EPOCHS, rollouts_per_epoch=ROLLOUTS_PER_EPOCH, render_every=10)
rollout.render(repeats=10)
