from trickster.agent import PPO
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.model import mlp
from trickster.utility import gymic

envs = [gymic.rwd_scaled_env() for _ in range(8)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = PPO(actor,
            critic,
            action_space=envs[0].action_space,
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.05)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, gymic.rwd_scaled_env())

rollout.fit(episodes=1000, updates_per_episode=4, steps_per_update=16, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=10)
