import numpy as np
from keras.models import Model
from keras import backend as K

from ..abstract import AgentBase
from ..experience import Experience
from ..utility.numeric import discount_reward


class A2C(AgentBase):

    def __init__(self,
                 actor: Model,
                 critic: Model,
                 actions,
                 memory: Experience,
                 reward_discount_factor=0.99,
                 state_preprocessor=None,
                 entropy_penalty_coef=0.005):

        super().__init__(actions, memory, reward_discount_factor, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.action_indices = np.arange(len(self.possible_actions))
        self.possible_actions_onehot = np.eye(len(self.possible_actions))
        self.entropy_penalty_coef = entropy_penalty_coef
        self._make_actor_train_function()

    def _make_actor_train_function(self):
        softmax = self.actor.output
        value = self.critic.output
        discounted_reward = K.placeholder(shape=[None], name="discounted_reward")
        action_onehot = K.placeholder(shape=[None, len(self.possible_actions)], name="action_onehot")

        action_probability = K.sum(action_onehot * softmax, axis=1)
        log_probability = K.log(action_probability)
        entropy = -K.sum(action_probability * log_probability) * self.entropy_penalty_coef
        advantage = discounted_reward - value
        policy_gradient_utility = -K.mean(advantage * log_probability)
        train_actor_inputs = [self.actor.input, self.critic.input, action_onehot, discounted_reward]
        train_actor_outputs = [policy_gradient_utility, entropy]
        train_actor_updates = self.actor.optimizer.get_updates(
            loss=policy_gradient_utility + entropy, params=self.actor.trainable_weights)

        train_actor = K.function(
            inputs=train_actor_inputs,
            outputs=train_actor_outputs,
            updates=train_actor_updates
        )

        self.actor_train_function = train_actor

    def sample(self, state, reward):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)
        return action

    def push_experience(self, final_state, final_reward, done=True):
        S = np.array(self.states)
        A = np.array(self.actions)
        R = np.array(self.rewards[1:] + [final_reward])
        dR = discount_reward(R, self.gamma)
        F = np.zeros(len(S), dtype=bool)
        F[-1] = done

        self.states = []
        self.actions = []
        self.rewards = []

        self.memory.remember(S, A, R, dR, F)

    def fit(self, batch_size=32, verbose=1, reset_memory=True):
        S, S_, A, R, dR, F = self.memory.sample(batch_size)
        assert len(S)

        S = self.preprocess(S)
        action_onehot = self.possible_actions_onehot[A]
        actor_utility, actor_entropy = self.actor_train_function([S, S, action_onehot, dR])

        S_ = self.preprocess(S_)
        value_next = np.squeeze(self.critic.predict(S_))
        bellman_target = value_next * self.gamma + R
        bellman_target[F] = R[F]
        mean_bellman_error = self.critic.train_on_batch(S, bellman_target)

        if reset_memory:
            self.memory.reset()
        return {"actor_utility": actor_utility,
                "actor_entropy": actor_entropy,
                "critic_loss": mean_bellman_error}
