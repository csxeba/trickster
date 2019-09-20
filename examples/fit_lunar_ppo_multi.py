from trickster.agent import PPO
from trickster.rollout import MultiRolling, Trajectory
from trickster.utility import gymic
from trickster.model import mlp

envs = [gymic.rwd_scaled_env("LunarLander-v2") for _ in range(32)]
test_env = gymic.rwd_scaled_env("LunarLander-v2")

input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = mlp.wide_mlp_actor_categorical(input_shape, num_actions, adam_lr=1e-4)
critic = mlp.wide_mlp_critic_network(input_shape, output_dim=1, adam_lr=1e-4)
agent = PPO(actor,
            critic,
            action_space=num_actions,
            ratio_clip_epsilon=0.2,
            training_epochs=10,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=1e-3)

rollout = MultiRolling(agent, envs)
test_rollout = Trajectory(agent, test_env)

rollout.fit(episodes=1000, updates_per_episode=16, steps_per_update=1, update_batch_size=64,
            testing_rollout=test_rollout, plot_curves=True)

test_rollout.render(repeats=10)
