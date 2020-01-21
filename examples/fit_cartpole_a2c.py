import gym

from trickster.agent import A2C
from trickster.rollout import Trajectory, RolloutConfig
from trickster.model import mlp

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions, actor_lr=1e-3, critic_lr=1e-3)

agent = A2C(actor,
            critic,
            action_space=env.action_space,
            discount_factor_gamma=0.99,
            gae_lambda=0.97)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=200))
rollout.fit(epochs=300, rollouts_per_epoch=1, render_every=0)
rollout.render(repeats=100)
