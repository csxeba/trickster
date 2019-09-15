from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.model import mlp
from trickster.utility import gymic

env = gymic.rwd_scaled_env()
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = A2C(actor,
            critic,
            action_space=env.action_space,
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.01)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, gymic.rwd_scaled_env())

rollout.fit(episodes=1000, updates_per_episode=64, step_per_update=1, testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=10)
