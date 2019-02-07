import numpy as np

from ..abstract import AgentBase
from ..experience import Experience


class PPO(AgentBase):

    def __init__(self,
                 actor,
                 critic,
                 actions,
                 memory: Experience,
                 reward_discount_factor_gamma=0.99,
                 gae_factor_lambda=0.95,
                 state_preprocessor=None):
        super().__init__(actions, memory, reward_discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.critic = critic

    def sample(self, state, reward):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.possible_actions, p=probabilities, size=1))

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)

        return action

    def push_experience(self, final_state, final_reward, done=True):
        pass

    def _fit_critic(self, S, critic_targets, verbose=0):
        loss = self.critic.train_on_batch(S, critic_targets)
        if verbose:
            print("Critic loss: {:.4f}".format(loss))
        return loss

    def _fit_actor(self, S, policy_targets, verbose=0):
        loss = self.actor.train_on_batch(S, policy_targets)
        if verbose:
            print("Actor loss: {:.4f}".format(loss))
        return loss

    def fit(self, batch_size=32, verbose=1, reset_memory=True):
        S, S_, A, R, dR, F = self.memory.sample(batch_size)
        assert len(S)

        S = self.preprocess(S)
        S_ = self.preprocess(S_)

        critic_targets = np.squeeze(self.critic.predict(S_)) * self.gamma + R
        critic_targets[F] = R[F]

        values = np.squeeze(self.critic.predict(S))
        advantages = dR - values
        probabilities = self.possible_actions_onehot[A]
        policy_targets = probabilities * advantages[..., None]

        critic_loss = self._fit_critic(S, critic_targets, verbose)
        actor_loss = self._fit_actor(S, policy_targets, verbose)

        if reset_memory:
            self.memory.reset()
        return {"actor_loss": actor_loss, "critic_loss": critic_loss}
