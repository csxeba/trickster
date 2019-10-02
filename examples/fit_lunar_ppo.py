import gym

from trickster.agent import PPO
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.model import mlp

env = gym.make("LunarLander-v2")
test_env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions, critic_lr=2e-4)

agent = PPO(actor,
            critic,
            action_space=num_actions,
            ratio_clip_epsilon=0.3,
            training_epochs=10,
            discount_factor_gamma=0.99)

rollout = Rolling(agent, env, RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, test_env, RolloutConfig(max_steps=300))

rollout.fit(episodes=1000, updates_per_episode=1, step_per_update=200, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True, render_every=100)
test_rollout.render(repeats=100)
