import numpy as np

from trickster.agent import TD3
from trickster.rollout import Trajectory, Rolling, RolloutConfig
from trickster.utility import spaces, gymic
from trickster.experience import Experience
from trickster.model import mlp

env = gymic.rwd_scaled_env("LunarLanderContinuous-v2", reward_scale=0.01)

input_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]

actor, critics = mlp.wide_ddpg_actor_critic(input_shape, output_dim=num_actions, action_range=2, num_critics=2,
                                            actor_lr=5e-4, critic_lr=5e-4)

assert critics[0] is not critics[1]

for layer1, layer2 in zip(critics[0].layers, critics[1].layers):
    assert layer1 is not layer2
    for w1, w2 in zip(layer1.weights, layer2.weights):
        assert w1 is not w2

for w_1, w_2 in zip(critics[0].get_weights(), critics[1].get_weights()):
    if w_1.ndim == 1:
        continue
    assert not np.all(w_1 == w_2)

agent = TD3(actor, critics,
            action_space=spaces.CONTINUOUS,
            memory=Experience(max_length=int(1e4)),
            discount_factor_gamma=0.99,
            action_noise_sigma=0.1,
            action_noise_sigma_decay=1.,
            action_minima=-2,
            action_maxima=2,
            target_noise_sigma=0.2,
            target_noise_clip=0.5)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, env, RolloutConfig(testing_rollout=True))

rollout.fit(episodes=1000, updates_per_episode=64, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout)
test_rollout.render(repeats=10)
