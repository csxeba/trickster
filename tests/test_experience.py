import unittest

import numpy as np

from trickster.experience import Experience


class TestExperienceConstructor(unittest.TestCase):

    def test_experience_constructor_considers_max_size_argument(self):
        xp = Experience(["state"], max_length=3)
        self.assertEqual(xp.max_length, 3)


class TestExperienceRemember(unittest.TestCase):

    def test_experience_remembers_vector_state(self):
        xp = Experience(["state", "reward", "done"])
        states = np.arange(10)
        rewards = np.arange(11)
        dones = np.random.random(size=11) < 0.5

        for state, reward, done in zip(states, rewards[:-1], dones[:-1]):
            xp.store_transition(state=state, reward=reward, done=done)

        xp.finalize_trajectory(states[-1], rewards[-1], dones[-1])

        self.assertEqual(xp.N, 10)
        np.testing.assert_equal(xp.memoirs["state"], states)
        np.testing.assert_equal(xp.memoirs["reward"], rewards[1:])
        np.testing.assert_equal(xp.memoirs["done"], dones[1:])
        np.testing.assert_equal(xp.final_state, states[-1])
