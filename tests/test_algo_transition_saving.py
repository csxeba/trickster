import unittest

import gym
import numpy as np
import tensorflow as tf

from trickster.agent import REINFORCE, DQN
from trickster.rollout import Rolling, Trajectory


class DummyEnv(gym.Env):

    ZERO = np.zeros(2, dtype="float32")

    def __init__(self):
        self.state = 1
        self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Box(0, 10, shape=[2])

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

        rollout.roll(STEPS, verbose=0, learning=True)

        state, logits, action, reward, done = agent.memory_sampler.sample(-1)

        self.assertEqual(agent.episodes, 3)
        np.testing.assert_array_less(state, 10)
        self.assertEqual(len(state), STEPS - 4)

    def test_dqn_doesnt_store_invalid_transitions(self):

        STEPS = 55

        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_dim=2)])
        model.compile(tf.keras.optimizers.SGD(learning_rate=0.), loss="categorical_crossentropy")
        agent = DQN(model, action_space=2, discount_gamma=0., use_target_network=False)
        rollout = Rolling(agent, DummyEnv())
        test_rollout = Trajectory(agent, DummyEnv())

        rollout.fit(epochs=10, updates_per_epoch=12, steps_per_update=STEPS, update_batch_size=8,
                    testing_rollout=test_rollout, plot_curves=False, render_every=0, warmup_buffer=False)

        state, state_next, action, reward, done = agent.memory_sampler.sample(-1)

        np.testing.assert_array_less(state, 10)
        np.testing.assert_equal((state_next - state).sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
