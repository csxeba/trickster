import gym

from trickster.agent import A2C
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("CartPole-v1")

agent = A2C.from_environment(env, discount_gamma=0.99, entropy_beta=0.01)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=200))
rollout.fit(epochs=300, rollouts_per_epoch=10, render_every=0)
rollout.render(repeats=100)
