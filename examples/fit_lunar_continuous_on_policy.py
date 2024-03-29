import gym

from trickster.agent import REINFORCE, A2C, PPO
from trickster.rollout import Trajectory
from trickster import callbacks

ENV_NAME = "LunarLanderContinuous-v2"
ALGO = "REINFORCE"
TRAJECTORY_MAX_STEPS = 100
EPOCHS = 1000
ROLLOUTS_PER_EPOCH = 4

env = gym.make(ENV_NAME)

algo = {"REINFORCE": REINFORCE,
        "A2C": A2C,
        "PPO": PPO}[ALGO]

agent = algo.from_environment(env)
rollout = Trajectory(agent, env, TRAJECTORY_MAX_STEPS)

cbs = [callbacks.ProgressPrinter(keys=rollout.progress_keys)]

rollout.fit(epochs=EPOCHS, updates_per_epoch=1, rollouts_per_update=4, callbacks=cbs)
rollout.render(repeats=100)
