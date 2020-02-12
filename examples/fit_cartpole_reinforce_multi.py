import gym

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory
from trickster.model import mlp

env = gym.make("CartPole-v1")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = mlp.wide_mlp_actor_categorical(input_shape, num_actions, adam_lr=1e-4)

agent = REINFORCE(policy, action_space=num_actions)

rollout = Trajectory(agent, env)
rollout.fit(episodes=1000, rollouts_per_update=8, update_batch_size=-1, plot_curves=True, render_every=100)
rollout.render(100)
