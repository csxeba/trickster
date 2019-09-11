import gym

from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.model import mlp


class CartPole(gym.RewardWrapper):

    def __init__(self):
        super().__init__(gym.make("CartPole-v1"))

    def reward(self, reward):
        return reward / 100


env = CartPole()
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions)

agent = A2C(actor,
            critic,
            action_space=env.action_space,
            absolute_memory_limit=10000,
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.01)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, CartPole())

rollout.fit(episodes=1000, updates_per_episode=32, step_per_update=32, testing_rollout=test_rollout, plot_curves=True)

test_rollout.render()
