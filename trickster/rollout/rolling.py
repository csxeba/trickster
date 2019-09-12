import numpy as np

from .trajectory import Trajectory
from .abstract import RolloutBase, RolloutConfig
from ..abstract import RLAgentBase
from ..utility import spaces, training_ops


class Rolling(RolloutBase):

    """Generate n-step trajectories for Time-Difference learning and infinite horizon problems"""

    def __init__(self, agent: RLAgentBase, env, config: RolloutConfig=None):
        super().__init__(agent, env, config)
        self.step = 0
        self.episodes = 0
        self.state = None
        self.action = None
        self.reward = None
        self.info = None
        self.done = None
        self.worker = agent.create_worker()
        self._rolling_worker = None

    def _sample_action(self):
        return self.worker.sample(self.state, self.reward, self.done)

    def _rolling_job(self):
        while 1:
            self._reset()
            while 1:
                yield self.step
                if self._finished():
                    break
                if self.step % self.cfg.skipframes == 0:
                    self.action = self._sample_action()
                assert not self.done
                if self.agent.action_space == spaces.CONTINUOUS:
                    a = self.action
                else:
                    a = self.agent.action_space[self.action]
                self.state, self.reward, self.done, self.info = self.env.step(a)
                self.step += 1

    def roll(self, steps, verbose=0, push_experience=True):
        if self._rolling_worker is None:
            self._rolling_worker = self._rolling_job()

        rewards = []

        self.worker.set_learning_mode(push_experience)
        for i, step in enumerate(self._rolling_worker):
            rewards.append(self.reward)
            if verbose:
                print("Step {} rwd: {:.4f}".format(self.step, self.reward))
            if i >= steps:
                break

        if push_experience:
            self.worker.push_experience(self.state, self.reward, self.done)

        self.worker.set_learning_mode(False)

        return {"mean_reward": np.mean(rewards), "rewards": np.array(rewards)}

    def _reset(self):
        self.reward = self.cfg.initial_reward
        self.info = {}
        self.done = False
        self.state = self.env.reset()
        self.step = 0
        self.episodes += 1

    def _finished(self):
        done = self.done
        if self.cfg.max_steps is not None:
            done = done or self.step >= self.cfg.max_steps
        return done

    def fit(self, episodes, updates_per_episode=32, step_per_update=32, testing_rollout: Trajectory=None,
            plot_curves=True):

        training_ops.fit(self, episodes, updates_per_episode, step_per_update, testing_rollout, plot_curves)
