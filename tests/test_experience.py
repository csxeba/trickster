import unittest

import numpy as np

from trickster.experience import Experience


class TestExperienceConstructor(unittest.TestCase):

    def test_experience_constructor_creates_empty_object(self):
        xp = Experience()

        self.assertIsNone(xp.memoirs)
        self.assertEqual(xp.N, 0)

    def test_experience_constructor_considers_max_size_argument(self):
        xp = Experience(max_length=3)

        self.assertEqual(xp.max_length, 3)


class TestExperienceRemember(unittest.TestCase):

    def test_experience_remembers_array(self):
        xp = Experience()
        xp.remember(np.arange(100))

        self.assertEqual(xp.N, 100)
        self.assertListEqual(xp.memoirs[0].tolist(), list(range(100)))

    def test_remember_considers_max_size(self):
        xp = Experience(max_length=100)
        xp.remember(np.arange(120))

        self.assertTrue(xp.N, 100)
        self.assertListEqual(xp.memoirs[0].tolist(), list(range(20, 120)))

    def test_remember_handles_multiple_arrays(self):
        xp = Experience()
        a = np.arange(100)
        b = a - 10
        c = a / 10

        xp.remember(a, b, c)

        self.assertEqual(len(xp.memoirs), 3)
        for source, target in zip(xp.memoirs, [a, b, c]):
            self.assertListEqual(source.tolist(), target.tolist())


class TestExperienceSample(unittest.TestCase):

    def setUp(self):
        self.xp = Experience(max_length=100)
        self.a = np.arange(100)
        self.b = self.a - 100
        self.c = self.a / 10

    def test_sampling_of_next_states(self):
        self.xp.remember(self.a)
        states, next_states = self.xp.sample(3)

        diff = next_states - states

        self.assertTrue(np.all(diff == 1))

    def test_sampling_when_samples_are_fewer_than_sample_size(self):
        self.xp.remember(self.a)
        states, next_states = self.xp.sample(200)

        self.assertTrue(len(states) == len(self.a) - 1)
        self.assertTrue(len(next_states) == len(self.a) - 1)

    def test_last_state_doesnt_get_sampled(self):
        self.xp.remember(self.a)
        states, next_states = self.xp.sample(200)

        self.assertNotIn(self.a[-1], states)

    def test_excluded_state_doesnt_get_sampled(self):
        EXCLUDE = (10, 20, 30)
        self.xp.remember(self.a, exclude=EXCLUDE)

        states, next_states = self.xp.sample(200)
        for x in EXCLUDE:
            self.assertNotIn(x-1, states)


if __name__ == '__main__':
    unittest.main()
