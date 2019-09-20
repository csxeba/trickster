from trickster.agent import DDPG
from trickster.model import mlp
from trickster.rollout import Rolling, Trajectory
from trickster.utility import gymic

env = gymic.rwd_scaled_env("Pendulum-v0", reward_scale=0.0006)
test_env = gymic.rwd_scaled_env("Pendulum-v0", reward_scale=0.0006)

actor, critic = mlp.wide_ddpg_actor_critic(env.observation_space.shape, env.action_space.shape[0],
                                           action_range=(-2, 2), actor_activation="tanh")

agent = DDPG(actor, critic, env.action_space, action_noise_sigma=0.1, action_minima=-2, action_maxima=2)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, env)

rollout.roll(steps=1024, push_experience=True)  # warmup
rollout.fit(episodes=500, updates_per_episode=32, step_per_update=1, update_batch_size=256, testing_rollout=test_rollout)
test_rollout.render(repeats=10)
