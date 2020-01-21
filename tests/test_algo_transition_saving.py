import unittest

import numpy as np
import tensorflow as tf

from trickster.agent import REINFORCE, DQN
from trickster.rollout import Rolling


class DummyEnv:

    ZERO = np.zeros(2)

    def __init__(self):
        self.state = 1

    def step(self, action):
        reward = self.state % 5 == 0
        done = self.state % 10 == 0
        obs = self.ZERO.copy()
        obs[1] = self.state
        self.state += 1
        return obs, reward, done, {}

    def reset(self):
        self.state = 1
        return self.ZERO


class TestTransitionValidity(unittest.TestCase):

    def test_reinforce_doesnt_store_invalid_transitions(self):

        STEPS = 35

        env = DummyEnv()
        actor = tf.keras.Sequential([tf.keras.layers.Dense(2, input_dim=2)])
        actor.compile(tf.keras.optimizers.SGD(learning_rate=0.), loss="categorical_crossentropy")
        agent = REINFORCE(actor, action_space=2, discount_factor_gamma=0.)
        rollout = Rolling(agent, env)

        rollout.roll(STEPS, verbose=1, learning=True)

        state, logits, action, reward, done = agent.memory_sampler.sample(-1)

        np.testing.assert_array_less(state, 10)
        self.assertEqual(len(state), STEPS - 4)

    def test_dqn_doesnt_store_invalid_transitions(self):

        STEPS = 35

        env = DummyEnv()
        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_dim=2)])
        model.compile(tf.keras.optimizers.SGD(learning_rate=0.), loss="categorical_crossentropy")
        agent = DQN(model, action_space=2, discount_factor_gamma=0.)
        rollout = Rolling(agent, env)

        rollout.roll(STEPS, verbose=1, learning=True)

        state, state_next, action, reward, done = agent.memory_sampler.sample(-1)

        np.testing.assert_array_less(state, 10)
        np.testing.assert_equal((state_next - state).sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
