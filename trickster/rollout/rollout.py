from ..abstract import AgentBase


class RolloutConfig:

    def __init__(self,
                 max_steps=None,
                 learning=True,
                 skipframes=None,
                 screen=None,
                 initial_reward=None):
        self.max_steps = max_steps
        self.learning = learning
        self.skipframes = skipframes or 1
        self.screen = screen
        self.initial_reward = initial_reward


class Rollout:

    def __init__(self, agent: AgentBase, env, config: RolloutConfig=None):
        self.env = env
        self.agent = agent
        self.cfg = config or RolloutConfig()
        self.done = False
        self.episodes = 0
        self.step = 0

        self.rolling = self._environment_coroutine()

    def _environment_coroutine(self):
        """Infinite loop ensures continuous rolling of agent inside the environment"""
        state = self.env.reset()
        reward = 0.
        info = {}
        action = self.agent.sample(state, reward=0.)
        while 1:
            yield state, reward, info
            if self.done:
                break
            state, reward, self.done, info = self.env.step(self.agent.possible_actions[action])
            action = self.agent.sample(state, reward)
            self.step += 1
            if self.cfg.screen is not None:
                self.cfg.screen.blit(state)

    def roll(self, steps, verbose=0, learning_batch_size=32):
        """Roll the agent inside the environment for <steps> steps for eg. TD-learning"""
        reward_sum = 0.
        state = None
        reward = None

        for i, (state, reward, info) in enumerate(self.rolling, start=1):
            if verbose:
                print("Step {} rwd: {:.4f}".format(self.step, reward))

            reward_sum += reward

            if i >= steps:
                break
            if self.finished:
                break

        history = {"reward_sum": reward_sum}

        if self.cfg.learning:
            self.agent.push_experience(state, reward, done=self.done)
            if learning_batch_size:
                result = self.agent.fit(batch_size=learning_batch_size, verbose=verbose)
                history.update(result)

        return history

    def rollout(self, verbose=1, learning_batch_size=0):
        """Generate a complete trajectory for eg. MCMC learning"""
        state = self.env.reset()
        reward = 0.
        cumulative_reward = 0.
        done = False
        step = 0
        action = None  # IDE bullies me heavily

        while not done:
            if self.cfg.screen is not None:
                self.cfg.screen.blit(state)
            if verbose:
                print("\rStep: {} total reward: {:.4f}".format(step+1, cumulative_reward), end="")
            if step % self.cfg.skipframes == 0:
                action = self.agent.sample(state, reward)
            state, reward, done, info = self.env.step(self.agent.possible_actions[action])
            cumulative_reward += reward
            step += 1
            if self.cfg.max_steps:
                if step >= self.cfg.max_steps:
                    break
        if verbose:
            print()
        history = {"reward_sum": cumulative_reward}
        if self.cfg.learning:
            self.agent.push_experience(state, reward, done)
            if learning_batch_size:
                result = self.agent.fit(batch_size=learning_batch_size, verbose=verbose)
                history.update(result)
        return history

    def reset(self):
        self.rolling = self._environment_coroutine()
        self.done = False
        self.episodes += 1
        self.step = 0

    @property
    def finished(self):
        result = self.done
        if self.cfg.max_steps is not None:
            result = result or self.step >= self.cfg.max_steps
        return result
