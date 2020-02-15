import unittest

import numpy as np

from trickster.experience import replay_buffer


class TestExperienceConstructor(unittest.TestCase):

    def test_experience_constructor_considers_max_size_argument(self):
        xp = replay_buffer.Experience(["state"], max_length=3)
        self.assertEqual(xp.max_length, 3)

    def test_experience_width_is_set_right(self):
        xp = replay_buffer.Experience("state, action, reward, done".split(", "))
        self.assertEqual(xp.width, 4)
        self.assertEqual(len(xp.memoirs), 4)


class TestExperienceRemember(unittest.TestCase):

    def test_experience_remembers_transition(self):
        xp = replay_buffer.Experience(["state", "reward", "done"])
        states = np.arange(10)
        rewards = np.arange(10)
        dones = np.random.random(size=10) < 0.5
        transition = replay_buffer.Transition(xp.keys)

        for state, reward, done in zip(states, rewards, dones):
            transition.set(state=state, reward=reward, done=done)
            xp.store(transition)

        self.assertEqual(xp.N, 10)

        np.testing.assert_equal(xp.memoirs["state"], states)
        np.testing.assert_equal(xp.memoirs["reward"], rewards)
        np.testing.assert_equal(xp.memoirs["done"], dones)

    def test_experience_remembers_dictionary_data(self):
        xp = replay_buffer.Experience(["state", "reward", "done"])
        states = np.arange(10)
        rewards = np.arange(10)
        dones = np.random.random(size=10) < 0.5

        xp.store(dict(state=states, reward=rewards, done=dones))

        self.assertEqual(xp.N, 10)

        np.testing.assert_equal(xp.memoirs["state"], states)
        np.testing.assert_equal(xp.memoirs["reward"], rewards)
        np.testing.assert_equal(xp.memoirs["done"], dones)

