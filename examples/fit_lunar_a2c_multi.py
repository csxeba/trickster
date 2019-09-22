from trickster.agent import A2C
from trickster.model import mlp
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.utility import gymic

NUM_ENVS = 32

envs = [gymic.rwd_scaled_env("LunarLander-v2", reward_scale=1) for _ in range(NUM_ENVS)]
test_env = gymic.rwd_scaled_env("LunarLander-v2", reward_scale=1)
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = A2C(actor,
            critic,
            action_space=num_actions,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.01)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=200))

rollout.fit(episodes=500, updates_per_episode=200, steps_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(100)
