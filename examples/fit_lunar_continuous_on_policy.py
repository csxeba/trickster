import gym

from trickster.agent import REINFORCE, A2C, PPO
from trickster.rollout import Trajectory

ENV_NAME = "LunarLanderContinuous-v2"
ALGO = "PPO"
TRAJECTORY_MAX_STEPS = 100
EPOCHS = 1000
ROLLOUTS_PER_EPOCH = 4

env = gym.make(ENV_NAME)

algo = {"REINFORCE": REINFORCE,
        "A2C": A2C,
        "PPO": PPO}[ALGO]

agent = algo.from_environment(env)
rollout = Trajectory(agent, env, TRAJECTORY_MAX_STEPS)

rollout.fit(epochs=EPOCHS, rollouts_per_epoch=ROLLOUTS_PER_EPOCH)
rollout.render(repeats=10)
