from trickster.agent import DDPG
from trickster.model import mlp
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.utility import gymic

env = gymic.rwd_scaled_env("Pendulum-v0", reward_scale=1.)
test_env = gymic.rwd_scaled_env("Pendulum-v0", reward_scale=1.)

actor, critic = mlp.wide_ddpg_actor_critic(env.observation_space.shape, env.action_space.shape[0],
                                           action_range=(-2, 2))

agent = DDPG(actor, critic, env.action_space, action_noise_sigma=0.1, action_minima=-2, action_maxima=2)

rollout = Rolling(agent, env, RolloutConfig(100))
test_rollout = Trajectory(agent, env, RolloutConfig(100))

rollout.fit(episodes=500, updates_per_episode=100, step_per_update=1, update_batch_size=128, plot_curves=True,
            testing_rollout=test_rollout, render_every=100)
