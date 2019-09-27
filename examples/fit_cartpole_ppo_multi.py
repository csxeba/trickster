from trickster.agent import PPO
from trickster.rollout import MultiRolling, Trajectory
from trickster.model import mlp
from trickster.utility import gymic

envs = [gymic.rwd_scaled_env("CartPole-v1") for _ in range(32)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = PPO(actor,
            critic,
            action_space=envs[0].action_space,
            discount_factor_gamma=0.98,
            ratio_clip_epsilon=0.3)

rollout = MultiRolling(agent, envs)
test_rollout = Trajectory(agent, gymic.rwd_scaled_env("CartPole-v1"))

rollout.fit(episodes=1000, updates_per_episode=16, steps_per_update=32, update_batch_size=64,
            testing_rollout=test_rollout, plot_curves=True, render_every=100)
test_rollout.render(repeats=100)
