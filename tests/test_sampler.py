import unittest

import numpy as np

from trickster.experience import Experience, ExperienceSampler


class TestExperienceSample(unittest.TestCase):

    def setUp(self):
        SIZE = 100
        self.xp = Experience(["state", "next_state", "done"], max_length=SIZE)
        self.sampler = ExperienceSampler(self.xp)
        self.states = np.arange(SIZE)
        self.final_state = SIZE
        self.dones = np.zeros(SIZE+1)

    def test_sampling_of_next_states(self):
        last_state = None
        for state, done in zip(self.states, self.dones[:-1]):
            if last_state is not None:
                self.xp.store_transition(state=last_state, next_state=state, done=done)
            last_state = state
        self.xp.finalize_trajectory(state=self.final_state, done=self.dones[-1])
        states, next_states, dones = self.sampler.sample(10)

        diff = next_states - states

        self.assertTrue(np.all(diff == 1))

    def test_sampling_when_samples_are_fewer_than_sample_size(self):
        SAMPLE_SIZE = 200
        self.xp.store_trajectory(self.states, dones=self.dones, final_state=self.final_state)
        states, next_states, dones = self.sampler.sample(SAMPLE_SIZE)

        self.assertTrue(len(states) == len(self.states))
        self.assertTrue(len(next_states) == len(self.states))

    def test_last_state_doesnt_get_sampled(self):
        self.xp.store_trajectory(self.states, dones=self.dones, final_state=self.final_state)
        states, next_states, dones = self.sampler.sample(200)

        self.assertNotIn(self.final_state, states)


class TestMultiExperienceSetup(unittest.TestCase):

    def setUp(self):
        self.num_samples = 10
        self.num_memories = 5

        a = np.arange(self.num_samples)
        b = -a
        dones = np.random.random(self.num_samples) < 0.5
        final = 10
        xps = [Experience() for _ in range(self.num_memories)]
        for i, xp in enumerate(xps, start=0):
            aa = a+10*i
            bb = b+10*i
            xp.store_trajectory(aa, bb, dones=dones, final_state=final)

        self.sampler = ExperienceSampler(xps)

    def test_width_is_calculated_correctly(self):
        self.assertEqual(4, self.sampler.width)

    def test_valid_indices_include_every_memory(self):
        result = self.sampler._get_valid_indices()
        target = np.array([[i, j] for i in range(self.num_memories) for j in range(self.num_samples)])
        self.assertTrue(np.all(target == result))

    def test_sampling_from_multiple_experiences(self):
        states, states_next, sample_b, dones = self.sampler.sample(size=25)
        self.assertEqual(25, len(states))


if __name__ == '__main__':
    unittest.main()
