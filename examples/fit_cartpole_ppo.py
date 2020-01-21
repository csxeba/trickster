import gym

from trickster.agent import PPO
from trickster.rollout import Trajectory, RolloutConfig
from trickster.model import mlp

env = gym.make("CartPole-v1")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions, actor_lr=1e-3, critic_lr=1e-3)

agent = PPO(env.action_space, actor, critic,
            discount_factor_gamma=0.99,
            gae_lambda=0.97,
            entropy_penalty_beta=0.01,
            ratio_clip_epsilon=0.3,
            target_kl_divergence=0.01,
            actor_updates=10,
            critic_updates=10)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=200))

rollout.fit(epochs=300, rollouts_per_epoch=1, update_batch_size=32, plot_curves=True, render_every=0)
rollout.render(repeats=100)
