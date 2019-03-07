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


if __name__ == '__main__':
    unittest.main()
