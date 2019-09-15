from trickster.agent import REINFORCE
from trickster.rollout import Trajectory, RolloutConfig
from trickster.utility import gymic
from trickster.model import mlp

env = gymic.rwd_scaled_env()
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = mlp.wide_mlp_actor_categorical(input_shape, num_actions, adam_lr=1e-4)
agent = REINFORCE(policy, action_space=num_actions)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))

rollout.fit(episodes=500, rollouts_per_update=1, update_batch_size=-1)
rollout.render(repeats=10)
