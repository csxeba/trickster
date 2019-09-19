from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory
from trickster.utility import gymic
from trickster.model import mlp

env = gymic.rwd_scaled_env("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = A2C(actor,
            critic,
            action_space=num_actions,
            absolute_memory_limit=10000,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.1)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, gymic.rwd_scaled_env("LunarLander-v2"))

rollout.fit(episodes=1000, updates_per_episode=64, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=10)
