import gym

from trickster.agent import REINFORCE, A2C, PPO
from trickster.rollout import Trajectory, RolloutConfig


ENV_NAME = "Pong-v0"
ALGO = "PPO"
TRAJECTORY_MAX_STEPS = None
EPOCHS = 1000
ROLLOUTS_PER_EPOCH = 1


class Pong(gym.ObservationWrapper):

    def __init__(self):
        super().__init__(env=gym.make(ENV_NAME))
        self.past = None

    @staticmethod
    def _process_frame(I):
        I = I[27:185]
        I = I[::2, ::2, 0].astype("float32")  # 80 x 80
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I

    def observation(self, I):
        I = self._process_frame(I)
        dif = self.past - I
        self.past = I
        return dif[..., None]

    def reset(self, **kwargs):
        I = self.env.reset(**kwargs)
        self.past = self._process_frame(I)
        return self.past[..., None]


env = Pong()

algo = {"REINFORCE": REINFORCE,
        "A2C": A2C,
        "PPO": PPO}[ALGO]

agent = algo.from_environment(env)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))
rollout.fit(epochs=EPOCHS, rollouts_per_epoch=ROLLOUTS_PER_EPOCH, render_every=100)
