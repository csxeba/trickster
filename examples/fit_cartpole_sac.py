import gym

from trickster.agent import SAC
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.model import mlp

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = SAC(actor,
            critic,
            action_space=env.action_space,
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.1)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env)

rollout.fit(epochs=1000, updates_per_epoch=16, step_per_update=4, testing_rollout=test_rollout, plot_curves=True,
            render_every=100)
test_rollout.render(repeats=100)
