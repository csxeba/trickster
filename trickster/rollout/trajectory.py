from ..abstract import AgentBase
from .abstract import RolloutBase, RolloutConfig


class Trajectory(RolloutBase):

    """Generate complete trajectories for MCMC learning or testing purposes"""

    def __init__(self, agent: AgentBase, env, config: RolloutConfig=None):
        super().__init__(agent, env, config)
        self.episodes = 0

    def rollout(self, verbose=1, push_experience=True, render=False):
        """Generate a complete trajectory for eg. MCMC learning"""
        self.agent.set_learning_mode(push_experience)
        state = self.env.reset()
        reward = self.cfg.initial_reward
        done = False
        reward_sum = 0.
        step = 0
        while not self._finished(done, step):
            step += 1
            if render:
                self.env.render()
            action = self.agent.sample(state, reward, done)
            state, reward, done, info = self.env.step(action)
            reward_sum += reward
            if verbose:
                print("\rStep: {} total reward: {:.4f}".format(step, reward_sum), end="")
        if verbose:
            print()
        if push_experience:
            self.agent.push_experience(state, reward, done)
        self.agent.set_learning_mode(not push_experience)

        return {"reward_sum": reward_sum}

    def _finished(self, current_done_value, current_step):
        done = current_done_value
        if self.cfg.max_steps is not None:
            done = done or current_step >= self.cfg.max_steps
        return done
