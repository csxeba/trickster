import unittest

import numpy as np

from trickster.experience import Experience, ExperienceSampler


class TestExperienceSample(unittest.TestCase):

    def setUp(self):
        self.xp = Experience(max_length=100)
        self.sampler = ExperienceSampler(self.xp)
        self.a = np.arange(100)
        self.b = self.a - 100
        self.c = self.a / 10

    def test_sampling_of_next_states(self):
        self.xp.remember(self.a)
        states, next_states = self.sampler.sample(10)

        diff = next_states - states

        self.assertTrue(np.all(diff == 1))

    def test_sampling_when_samples_are_fewer_than_sample_size(self):
        self.xp.remember(self.a)
        states, next_states = self.sampler.sample(200)

        self.assertTrue(len(states) == len(self.a) - 1)
        self.assertTrue(len(next_states) == len(self.a) - 1)

    def test_last_state_doesnt_get_sampled(self):
        self.xp.remember(self.a)
        states, next_states = self.sampler.sample(200)

        self.assertNotIn(self.a[-1], states)

    def test_excluded_state_doesnt_get_sampled(self):
        EXCLUDE = (10, 20, 30)
        self.xp.remember(self.a, exclude=EXCLUDE)

        states, next_states = self.sampler.sample(200)
        for x in EXCLUDE:
            self.assertNotIn(x, states)

    def test_excluding_negative_index_is_correctly_interpreted(self):
        EXCLUDE = (-1, -10, -20)
        TARGET = (99, 90, 80)
        self.xp.remember(self.a, exclude=EXCLUDE)

        states, next_states = self.sampler.sample(200)
        for t in TARGET:
            self.assertNotIn(t, states)

    def test_excluding_works_after_multiple_remembers(self):
        EXCLUDE = (10, 20, 30)
        for _ in range(3):
            self.xp.remember(self.a, exclude=EXCLUDE)

        states, next_states = self.sampler.sample(-1)
        for e in EXCLUDE:
            self.assertNotIn(e, states)


class TestMultiExperienceSetup(unittest.TestCase):

    def setUp(self):
        a = np.arange(10)
        b = -a
        xps = [Experience() for _ in range(5)]
        for i, xp in enumerate(xps, start=0):
            aa = a+10*i
            bb = b+10*i
            xp.remember(aa, bb)
        self.sampler = ExperienceSampler(xps)

    def test_width_is_calculated_correctly(self):
        self.assertEqual(2, self.sampler.width)

    def test_valid_indices_include_every_memory(self):
        result = self.sampler._get_valid_indices()
        target = np.array([[i, j] for i in range(5) for j in range(9)])
        self.assertTrue(np.all(target == result))

    def test_sampling_from_multiple_experiences(self):
        states, states_next, sample_b = self.sampler.sample(size=25)
        self.assertEqual(25, len(states))

    def test_samples_exclude_last_states(self):
        states, states_next, sample_b = self.sampler.sample(size=-1)
        EXCLUDE_STATE = [9, 19, 29, 39, 49]
        EXCLUDE_B = [-9, 1, 11, 21, 31]
        for e_s, e_b in zip(EXCLUDE_STATE, EXCLUDE_B):
            self.assertNotIn(e_s, states)
            self.assertNotIn(e_b, sample_b)


class TestMultiExperienceExclusion(unittest.TestCase):

    def test_sampling_considers_explicit_exclusions(self):
        exclusion = (3, 6)
        excluded_a = []
        excluded_b = []

        exps = []
        for i in range(1, 6):
            a = np.arange(10*i, 10*(i+1))
            b = a + 100
            exps.append(Experience())
            exps[-1].remember(a, b, exclude=exclusion)
            for e in exclusion:
                excluded_a.append(a[e])
                excluded_b.append(b[e])

        sampler = ExperienceSampler(exps)
        states, states_next, sample_b = sampler.sample(-1)

        for e_a, e_b in zip(excluded_a, excluded_b):
            self.assertNotIn(e_a, states)
            self.assertNotIn(e_b, sample_b)


if __name__ == '__main__':
    unittest.main()
