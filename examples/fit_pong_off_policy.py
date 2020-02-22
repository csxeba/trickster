import gym

from trickster.agent import DQN, DoubleDQN
from trickster.rollout import Trajectory, Rolling, RolloutConfig

ENV = "Pong-v0"
ALGO = "DQN"
TRAJECTORY_MAX_STEPS = None
STEPS_PER_UPDATE = 1
# WARMUP_STEPS = int(1e4)
WARMUP_STEPS = 1000
MEMORY_BUFFER_SIZE = int(1e5)
UPDATES_PER_EPOCH = 64
EPOCHS = 300
UPDATE_BATCH_SIZE = 64


class Pong(gym.ObservationWrapper):

    def __init__(self):
        super().__init__(env=gym.make(ENV))
        self.past = None

    @staticmethod
    def _process_frame(I):
        I = I[27:185]
        I = I[::2, ::2, 0]  # 80 x 80
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
test_env = Pong()

algo = {"DQN": DQN,
        "DoubleDQN": DoubleDQN}[ALGO]

agent = algo.from_environment(env)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))

rollout.roll(WARMUP_STEPS, verbose=1, learning=True)
rollout.fit(epochs=EPOCHS, updates_per_epoch=UPDATES_PER_EPOCH, steps_per_update=STEPS_PER_UPDATE,
            update_batch_size=UPDATE_BATCH_SIZE,
            testing_rollout=test_rollout, plot_curves=True, render_every=0, warmup_buffer=True)

test_rollout.render(repeats=10)
