from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.utility import gymic
from trickster.model import mlp

env = gymic.rwd_scaled_env("LunarLander-v2", reward_scale=1.)
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions, actor_lr=1e-4, critic_lr=1e-4)

agent = A2C(actor,
            critic,
            action_space=num_actions,
            absolute_memory_limit=10000,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.001)

rollout = Rolling(agent, env, RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, gymic.rwd_scaled_env("LunarLander-v2"))

rollout.fit(episodes=100, updates_per_episode=200, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=5)
rollout.fit(episodes=100, updates_per_episode=200, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=5)
rollout.fit(episodes=100, updates_per_episode=200, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=5)
rollout.fit(episodes=100, updates_per_episode=200, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=5)
rollout.fit(episodes=100, updates_per_episode=200, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=5)
