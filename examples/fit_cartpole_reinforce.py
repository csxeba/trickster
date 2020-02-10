import gym

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("CartPole-v1")

agent = REINFORCE.from_environment(env, discount_gamma=0.99)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=200))

rollout.fit(epochs=300, rollouts_per_epoch=10, render_every=0)
rollout.render(repeats=10)
