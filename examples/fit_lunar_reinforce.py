import keras

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory
from trickster.experience import Experience
from trickster.utility import gymic
from trickster.model import mlp

env = gymic.rwd_scaled_env("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = mlp.wide_mlp_actor_categorical(input_shape, num_actions)
policy.compile(optimizer=keras.optimizers.SGD(lr=2e-4, momentum=0.9), loss="categorical_crossentropy")

agent = REINFORCE(policy,
                  action_space=num_actions,
                  memory=Experience(max_length=10000),
                  discount_factor_gamma=0.99)

rollout = Trajectory(agent, env)
rollout.fit(episodes=1000, rollouts_per_update=16, update_batch_size=-1)
rollout.render(repeats=10)
