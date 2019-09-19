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

    def test_experience_remembers_vector_state(self):
        xp = Experience()
        states = np.arange(10)
        dones = np.random.random(size=10) < 0.5
        final_state = 10

        xp.remember(states, dones=dones, final_state=final_state)

        next_state_target = np.append(states[1:], final_state)

        self.assertEqual(xp.N, 10)
        self.assertTrue(np.all(xp.memoirs[0] == states))
        self.assertTrue(np.all(xp.memoirs[1] == next_state_target))

    def test_experience_remembers_matrix_state(self):
        xp = Experience()
        states = np.repeat(np.arange(10), 10).reshape(10, 10)
        dones = np.random.random(size=10) < 0.5
        final_state = np.full(10, 10)
        xp.remember(states, dones=dones, final_state=final_state)

        self.assertEqual(xp.N, 10)
        for i in range(10):
            state_target = np.full(10, i)
            next_state_target = np.full(10, i+1)
            self.assertListEqual(xp.memoirs[0][i].tolist(), state_target.tolist())
            self.assertListEqual(xp.memoirs[1][i].tolist(), next_state_target.tolist())

    def test_remember_considers_max_size(self):
        SIZE = 12
        MAXSIZE = 10
        STATEDIM = 10

        xp = Experience(max_length=MAXSIZE)
        states = np.repeat(np.arange(SIZE), STATEDIM).reshape(SIZE, STATEDIM)
        dones = np.random.random(size=SIZE) < 0.5
        final_state = np.full(STATEDIM, SIZE)
        xp.remember(states, final_state=final_state, dones=dones)

        self.assertTrue(xp.N, MAXSIZE)
        for i in range(MAXSIZE):
            state = xp.memoirs[0][i]
            state_next = xp.memoirs[1][i]
            state_target = np.full(STATEDIM, i+2)
            next_state_target = np.full(STATEDIM, i+3)
            self.assertListEqual(state.tolist(), state_target.tolist())
            self.assertListEqual(state_next.tolist(), next_state_target.tolist())

    def test_remember_handles_multiple_arrays(self):
        SIZE = 10
        DIM = 10

        xp = Experience()
        state = np.repeat(np.arange(SIZE), DIM).reshape(SIZE, DIM)
        final = np.full(SIZE, SIZE)
        state_next = np.concatenate([state[1:], final[None, ...]], axis=0)
        dones = np.random.random(size=SIZE) < 0.5
        b = state[:, 0] - 10
        c = state[:, 0] / 10

        targets = state, state_next, b, c, dones

        xp.remember(state, b, c, final_state=final, dones=dones)

        self.assertEqual(xp.width, len(targets))
        for source, target in zip(xp.memoirs, targets):
            self.assertTrue(np.all(source == target))


if __name__ == '__main__':
    unittest.main()
