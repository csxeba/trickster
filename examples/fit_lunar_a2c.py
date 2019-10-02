import gym

from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.model import mlp

env = gym.make("LunarLander-v2")
test_env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions, actor_lr=1e-4, critic_lr=1e-4)

agent = A2C(actor,
            critic,
            action_space=num_actions,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.01)

rollout = Rolling(agent, env, RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env)

rollout.fit(episodes=2000, updates_per_episode=200, step_per_update=1, update_batch_size=-1,
            testing_rollout=test_rollout, plot_curves=True, render_every=100)
test_rollout.render(repeats=100)
