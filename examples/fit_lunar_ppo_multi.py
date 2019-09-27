from trickster.agent import PPO
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.utility import gymic
from trickster.model import mlp

NUM_ENVS = 32

envs = [gymic.rwd_scaled_env("LunarLander-v2", reward_scale=1) for _ in range(NUM_ENVS)]
test_env = gymic.rwd_scaled_env("LunarLander-v2", reward_scale=1)
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = PPO(actor,
            critic,
            action_space=num_actions,
            ratio_clip_epsilon=0.3,
            training_epochs=10,
            discount_factor_gamma=0.99)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=200))

rollout.fit(episodes=1000, updates_per_episode=1, steps_per_update=8, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True, render_every=100)
test_rollout.render(100)
