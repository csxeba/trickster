from ..abstract import AgentBase


class RolloutConfig:

    def __init__(self,
                 max_steps=None,
                 skipframes=None,
                 initial_reward=None):

        self.max_steps = max_steps
        self.skipframes = skipframes or 1
        self.initial_reward = initial_reward or 0.


class Rollout:

    def __init__(self, agent: AgentBase, env, config: RolloutConfig=None):
        self.env = env
        self.agent = agent
        self.cfg = config or RolloutConfig()  # type: RolloutConfig
        self.step = 0
        self.episodes = 0
        self.state = None
        self.action = None
        self.reward = None
        self.info = None
        self.done = None
        self._rolling_worker = None
        self._resetted = False

    def _sample_action(self):
        return self.agent.sample(self.state, self.reward, self.done)

    def _rolling_job(self, infinite=False):
        while 1:
            self._reset(infinite_mode=infinite)
            while 1:
                yield self.step
                if self.finished:
                    break
                if self.step % self.cfg.skipframes == 0:
                    self.action = self._sample_action()
                assert not self.done
                self.state, self.reward, self.done, self.info = self.env.step(
                    self.agent.possible_actions[self.action])
                self.step += 1
            if not infinite:
                break
        self._rolling_worker = None

    def roll(self, steps, verbose=0, push_experience=True):
        """Roll the agent inside the environment for <steps> steps for eg. TD-learning"""
        if self._rolling_worker is None:
            self._reset(infinite_mode=True)
        history = {"rewards": [], "reward_sum": 0.}
        self.agent.set_learning_mode(push_experience)
        for i, step in enumerate(self._rolling_worker):
            history["rewards"].append(self.reward)
            history["reward_sum"] += self.reward
            if verbose:
                print("Step {} rwd: {:.4f}".format(self.step, self.reward))
            if i >= steps:
                break

        if push_experience:
            self.agent.push_experience(self.state, self.reward, self.done)

        self.agent.set_learning_mode(not push_experience)

        return history

    def rollout(self, verbose=1, push_experience=True):
        """Generate a complete trajectory for eg. MCMC learning"""
        self._reset(infinite_mode=False)
        self.agent.set_learning_mode(push_experience)
        history = {"rewards": [], "reward_sum": 0.}
        for step in self._rolling_worker:
            history["rewards"].append(self.reward)
            history["reward_sum"] += self.reward
            if verbose:
                print("\rStep: {} total reward: {:.4f}".format(step+1, history["reward_sum"]), end="")

        if verbose:
            print()
        if push_experience:
            self.agent.push_experience(self.state, self.reward, self.done)

        self.agent.set_learning_mode(not push_experience)

        return history

    def _reset(self, infinite_mode):
        self.reward = self.cfg.initial_reward
        self.info = {}
        self.done = False
        self.state = self.env.reset()
        self.step = 0
        self.episodes += 1
        self._resetted = True
        if self._rolling_worker is None or not infinite_mode:
            self._rolling_worker = self._rolling_job(infinite=infinite_mode)

    @property
    def finished(self):
        done = self.done
        if self.cfg.max_steps is not None:
            done = done or self.step >= self.cfg.max_steps
        return done
