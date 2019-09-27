from trickster.agent import DDPG
from trickster.rollout import Trajectory, Rolling, RolloutConfig
from trickster.utility import spaces, gymic
from trickster.experience import Experience
from trickster.model import mlp

env = gymic.rwd_scaled_env("LunarLanderContinuous-v2")

input_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]

actor, critic = mlp.wide_ddpg_actor_critic(input_shape, output_dim=num_actions, action_range=2)

agent = DDPG(actor, critic,
             action_space=spaces.CONTINUOUS,
             memory=Experience(max_length=int(1e4)),
             discount_factor_gamma=1.,
             action_noise_sigma=0.1,
             action_noise_sigma_decay=0.99,
             action_minima=-2,
             action_maxima=2)

rollout = Rolling(agent, env, RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, env)

rollout.fit(episodes=10000, updates_per_episode=200, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout)
test_rollout.render(repeats=10)
