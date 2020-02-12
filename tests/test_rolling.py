import numpy as np

import unittest

from trickster.experience import Experience
from trickster.agent.abstract import RLAgentBase
from trickster.rollout import Rolling


class _DummyEnv:

    def __init__(self, pointer_start=0):
        self.states = np.arange(10)
        self.rewards = np.arange(10)
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
        done = self.pointer == self.states[-1]
        state, reward, done, info = self.states[self.pointer], self.rewards[self.pointer], done, {}
        self.states_played.append(state)
        self.rewards_played.append(reward)
        self.dones_played.append(done)
        return state, reward, done, info

    def step(self, action=None):
        self.pointer += 1
        return self.generate_data()


class _DummyAgent(RLAgentBase):

    transition_memory_keys = ["state", "state_next", "action", "reward", "done"]

    def __init__(self):
        super().__init__(action_space=10, training_memory=Experience(self.memory_keys, max_length=100))

    def sample(self, state, reward, done):
        action = state % 10
        self._push_step_to_direct_memory_if_learning(
            state=state, reward=reward, done=done)
        return action

    def get_savables(self):
        return {}

    def fit(self, batch_size=None):
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

        rollout.roll(steps=STEPS, learning=True)

        self.assertEqual(agent.transition_memory.N, 1)
        self.assertEqual(env.pointer, STEPS)

    def test_rolling_for_multiple_steps(self):
        STEPS = 5

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, learning=True)

        self.assertEqual(agent.transition_memory.N, STEPS)
        self.assertEqual(env.pointer, STEPS)

    def test_agent_memory_is_reseted(self):
        STEPS = 5

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, learning=True)

        mem = agent.transition_memory
        mem.reset()

        for key in mem.keys:
            self.assertIs(mem.memoirs[key], None)

    def test_agent_memory_is_pushed(self):
        STEPS = 5

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, learning=True)

        np.testing.assert_array_equal(agent.transition_memory.memoirs["state"], np.arange(STEPS))
        np.testing.assert_array_equal(agent.transition_memory.memoirs["reward"], np.arange(STEPS))
        np.testing.assert_array_equal(agent.transition_memory.memoirs["done"], np.zeros(STEPS, dtype=bool))

    def test_everything_is_stored_when_done_flag_is_reached(self):
        STEPS = 12

        agent = _DummyAgent()
        env = _DummyEnv()
        rollout = Rolling(agent, env)

        rollout.roll(steps=STEPS, learning=True)

        dones = np.zeros(STEPS, dtype=bool)
        dones[9] = True

        np.testing.assert_array_equal(dones, agent.transition_memory.memoirs["done"])
        self.assertEqual(2, env.pointer)


if __name__ == '__main__':
    unittest.main()
