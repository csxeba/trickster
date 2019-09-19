import numpy as np

import unittest

from trickster.experience import Experience
from trickster.abstract import RLAgentBase
from trickster.rollout import Rolling


class _DummyEnv:

    def __init__(self, pointer_start=0):
        self.states = np.arange(20)
        self.rewards = np.arange(20)
        self.pointer_start = pointer_start
        self.pointer = pointer_start
        self.states_played = []
        self.rewards_played = []
        self.dones_played = []

    def reset(self):
        self.pointer = self.pointer_start
        state = self.states[self.pointer]
        self.states_played.append(state)
        return state

    def generate_data(self):
        done = self.pointer == 10
        state, reward, done, info = self.states[self.pointer], self.rewards[self.pointer], done, {}
        self.states_played.append(state)
        self.rewards_played.append(reward)
        self.dones_played.append(done)
        return state, reward, done, info

    def step(self, action=None):
        self.pointer += 1
        return self.generate_data()


class _DummyAgent(RLAgentBase):

    def __init__(self):
        super().__init__(action_space=10, memory=Experience(max_length=100))

    def sample(self, state, reward, done):
        action = state
        self._push_step_to_direct_memory_if_learning(state, action, reward, done)
        return action

    def get_savables(self):
        return {}


class _DummyAgentWithWorker(_DummyAgent):

    def create_worker(self, **worker_kwargs):
        return None


class TestRollingRollout(unittest.TestCase):

    def test_rolling_constructor(self):
        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        self.assertIs(rollout.agent, agent)
        self.assertIs(rollout.env, env)

    def test_rolling_constructor_extracts_default_worker(self):
        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        self.assertIs(rollout.worker, agent)

    def test_rolling_constructor_extracts_explicit_worker(self):
        agent = _DummyAgentWithWorker()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        self.assertIs(rollout.worker, None)

    def test_rolling_one_step(self):
        STEPS = 1

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, push_experience=True)

        self.assertEqual(agent.memory.N, 1)
        self.assertEqual(env.pointer, STEPS)

    def test_rolling_for_multiple_steps(self):
        STEPS = 5

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, push_experience=True)

        self.assertEqual(agent.memory.N, STEPS)
        self.assertEqual(env.pointer, STEPS)

    def test_agent_memory_is_reseted(self):
        STEPS = 5

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, push_experience=True)

        self.assertListEqual(agent.states, [])
        self.assertListEqual(agent.dones, [])
        self.assertListEqual(agent.rewards, [])
        self.assertListEqual(agent.actions, [])

    def test_agent_memory_is_pushed(self):
        STEPS = 5

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, push_experience=True)

        np.testing.assert_array_equal(agent.memory.memoirs[0], np.arange(STEPS))
        np.testing.assert_array_equal(agent.memory.memoirs[1], np.arange(1, STEPS+1))
        np.testing.assert_array_equal(agent.memory.memoirs[2], np.arange(STEPS))
        np.testing.assert_array_equal(agent.memory.memoirs[4], np.zeros(STEPS, dtype=bool))

    def test_environment_is_reset_correctly(self):
        STEPS = 11

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, push_experience=True)

        dones = np.zeros(STEPS-1, dtype=bool)
        dones[-1] = True

        np.testing.assert_array_equal(dones, agent.memory.memoirs[-1])
        self.assertEqual(0, env.pointer)

    def test_memory_is_pushed_correctly_for_multiple_resets(self):
        """    0 0 0 0 0 0 0 0 0 0 1 1 1 1 1
        step   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
        ------------------------------------
        state  s S S S S s S S S S S s S S S
        ------------------------------------
        pstate S S S S   S S S S S   S S S S
        nstate   S S S S   S S S S S   S S S
        action A A A A   A A A A A   A A A A
        ------------------------------------
        reward   R R R R   R R R R R   R R R
        done     d d d D   d d d d D   d d d
        spoint + + + + - + + + + + - + + + +
        """
        STEPS = 27

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, verbose=0, push_experience=True)

        dones = agent.memory.memoirs[-1]
        states_remembered, next_states_remembered = agent.memory.memoirs[:2]
        states_diff = next_states_remembered - states_remembered

        np.testing.assert_array_equal(states_diff, np.ones(agent.memory.N))
        self.assertEqual(sum(dones), 2)
        self.assertNotIn(10, states_remembered)


if __name__ == '__main__':
    unittest.main()
