import unittest

import gym
import numpy as np
import tensorflow as tf

from trickster.agent import REINFORCE, A2C, DQN, TD3
from trickster.rollout import Rolling, Trajectory
from trickster.model import arch


class DummyEnv(gym.Env):

    ZERO = np.zeros(2, dtype="float32")

    def __init__(self, action_space="discreete"):
        self.state = 1
        if action_space == "discreete":
            self.action_space = gym.spaces.Discrete(n=2)
        elif action_space == "continuous":
            self.action_space = gym.spaces.Box(low=-1, high=+1, shape=[2])
        self.observation_space = gym.spaces.Box(0, 100, shape=[2])

    def step(self, action):
        if self.state > 10:
            raise RuntimeError
        reward = float(self.state % 5 == 0)
        done = self.state % 10 == 0
        obs = self.ZERO.copy()
        obs[1] = self.state
        self.state += 1
        return obs, reward, done, {}

    def reset(self):
        self.state = 1
        return self.ZERO

    def render(self, mode='human'):
        pass


class TestTransitionValidity(unittest.TestCase):

    def test_reinforce_doesnt_store_invalid_transitions(self):

        STEPS = 35

        env = DummyEnv()
        agent = REINFORCE.from_environment(env, discount_gamma=0.)
        rollout = Rolling(agent, env)

        rollout.roll(STEPS, verbose=0, push_experience=True)

        data = agent.memory_sampler.sample(-1)

        self.assertEqual(agent.episodes, 3)
        np.testing.assert_array_less(data["state"], 10)
        self.assertEqual(len(data["state"]), STEPS - 4)

    def test_a2c_doesnt_store_invalid_transitions_in_td_setting(self):

        STEPS = 35

        env = DummyEnv()
        agent = A2C.from_environment(env, discount_gamma=0.)
        rollout = Rolling(agent, env)

        rollout.roll(STEPS, verbose=0, push_experience=True)

        data = agent.memory_sampler.sample(-1)

        self.assertEqual(agent.episodes, 3)
        np.testing.assert_array_less(data["state"], 10)
        self.assertEqual(len(data["state"]), STEPS - 4)

    def test_dqn_doesnt_store_invalid_transitions(self):

        STEPS = 55

        env = DummyEnv()
        test_env = DummyEnv()

        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_dim=2)])
        model.compile(tf.keras.optimizers.SGD(learning_rate=0.), loss="mse")

        agent = DQN.from_environment(env, model, discount_gamma=0., use_target_network=False)

        rollout = Rolling(agent, env)
        test_rollout = Trajectory(agent, test_env)

        rollout.fit(epochs=10, updates_per_epoch=12, steps_per_update=STEPS, update_batch_size=8,
                    testing_rollout=test_rollout, buffer_warmup=False)

        data = agent.memory_sampler.sample(-1)

        np.testing.assert_array_less(data["state"], 10)
        np.testing.assert_equal((data["state_next"] - data["state"]).sum(axis=1), 1)

    def test_td3_doesnt_store_invalid_transitions(self):

        STEPS = 55

        env = DummyEnv(action_space="continuous")
        test_env = DummyEnv(action_space="continuous")

        actor_out = tf.convert_to_tensor([[0., 0.]])
        critic_out = tf.convert_to_tensor([1.])

        actor = arch.TestingModel(actor_out)
        critic = arch.TestingModel(critic_out)
        critic2 = arch.TestingModel(critic_out)

        actor_target = arch.TestingModel(actor_out)
        critic_target = arch.TestingModel(critic_out)
        critic2_target = arch.TestingModel(critic_out)

        actor.optimizer = tf.keras.optimizers.SGD(0)
        critic.optimizer = tf.keras.optimizers.SGD(0)
        critic2.optimizer = tf.keras.optimizers.SGD(0)

        agent = TD3(actor, actor_target, critic, critic_target, critic2, critic2_target,
                    discount_gamma=0., polyak_tau=0., action_minima=-1., action_maxima=1., update_actor_every=1)

        rollout = Rolling(agent, env)
        test_rollout = Trajectory(agent, test_env)

        rollout.fit(epochs=10, updates_per_epoch=12, steps_per_update=STEPS, update_batch_size=8,
                    testing_rollout=test_rollout, buffer_warmup=False)

        data = agent.memory_sampler.sample(-1)

        np.testing.assert_array_less(data["state"], 10)
        np.testing.assert_equal((data["state_next"] - data["state"]).sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
