import gym

from trickster.agent import TD3
from trickster.model import mlp
from trickster.rollout import Rolling, Trajectory, RolloutConfig

env = gym.make("Pendulum-v0")
test_env = gym.make("Pendulum-v0")

actor, critic = mlp.wide_ddpg_actor_critic(env.observation_space.shape, env.action_space.shape[0],
                                           action_range=(-2, 2), num_critics=2, actor_lr=1e-3, critic_lr=1e-3)

agent = TD3(actor, critic, env.action_space,
            action_noise_sigma=0.1,
            action_minima=-2,
            action_maxima=2,
            target_noise_sigma=0.2,
            target_noise_clip=0.5)

rollout = Rolling(agent, env, RolloutConfig(100))
test_rollout = Trajectory(agent, env, RolloutConfig(100))

rollout.fit(episodes=1500, updates_per_episode=100, step_per_update=1, update_batch_size=128,
            testing_rollout=test_rollout, plot_curves=True, render_every=100)
