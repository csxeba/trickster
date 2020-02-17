import gym

from trickster.agent import REINFORCE, A2C, PPO
from trickster.rollout import Trajectory, RolloutConfig

ENV_NAME = "CartPole-v1"
ALGO = "REINFORCE"
TRAJECTORY_MAX_STEPS = 200
EPOCHS = 300
ROLLOUTS_PER_EPOCH = 10

env = gym.make(ENV_NAME)

algo = {"REINFORCE": REINFORCE,
        "A2C": A2C,
        "PPO": PPO}[ALGO]

agent = algo.from_environment(env)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))

rollout.fit(epochs=EPOCHS, rollouts_per_epoch=ROLLOUTS_PER_EPOCH, render_every=0)
rollout.render(repeats=10)
