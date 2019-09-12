import gym

from trickster.agent import DQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling
from trickster.experience import Experience
from trickster.model import mlp


class CartPole(gym.RewardWrapper):

    def __init__(self):
        super().__init__(gym.make("CartPole-v1"))

    def reward(self, reward):
        return reward / 100


env = CartPole()
input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = mlp.wide_dueling_q_network(input_shape, num_actions, adam_lr=1e-3)

agent = DQN(ann,
            action_space=env.action_space,
            memory=Experience(max_length=10000),
            epsilon=1.,
            epsilon_decay=0.99995,
            epsilon_min=0.1,
            discount_factor_gamma=0.98,
            use_target_network=True)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, CartPole())

rollout.fit(episodes=500, updates_per_episode=128, step_per_update=2, testing_rollout=test_rollout, plot_curves=True)
test_rollout.render()
