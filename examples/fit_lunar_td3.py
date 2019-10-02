import gym

from trickster.agent import TD3
from trickster.rollout import Trajectory, Rolling
from trickster.utility import spaces
from trickster.experience import Experience
from trickster.model import mlp

env = gym.make("LunarLanderContinuous-v2")
test_env = gym.make("LunarLanderContinuous-v2")

input_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]

actor, critics = mlp.wide_ddpg_actor_critic(input_shape, output_dim=num_actions, action_range=2, num_critics=2)

agent = TD3(actor, critics,
            action_space=spaces.CONTINUOUS,
            memory=Experience(max_length=int(1e4)),
            discount_factor_gamma=0.99,
            action_noise_sigma=1.,
            action_noise_sigma_decay=0.99999,
            action_minima=-2,
            action_maxima=2,
            target_noise_sigma=0.2,
            target_noise_clip=0.5)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, test_env)

rollout.fit(episodes=1000, updates_per_episode=64, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, render_every=100)
test_rollout.render(repeats=100)
