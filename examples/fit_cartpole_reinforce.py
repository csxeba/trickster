import gym

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory, RolloutConfig
from trickster.model import mlp

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = mlp.wide_mlp_actor_categorical(input_shape, num_actions, adam_lr=1e-3)
agent = REINFORCE(policy, action_space=env.action_space, discount_factor_gamma=0.99)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=200))

rollout.fit(epochs=300, rollouts_per_epoch=1, render_every=0)
rollout.render(repeats=10)
