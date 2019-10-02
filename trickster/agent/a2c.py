from typing import List

import numpy as np
from tensorflow import keras

from ..abstract import RLAgentBase
from ..experience import Experience, ExperienceSampler
from ..utility import kerasic


class Actor(RLAgentBase):

    def __init__(self,
                 model: keras.Model,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.model = model
        self.action_indices = np.arange(len(self.action_space))
        self.possible_actions_onehot = np.eye(len(self.action_space))

    def sample(self, state, reward, done):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.model.predict(preprocessed_state)[0]
        if self.learning:
            action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))
        else:
            action = np.squeeze(np.argmax(probabilities, axis=-1))
        self._push_step_to_direct_memory_if_learning(state, action, reward, done)
        return action

    def get_savables(self):
        raise RuntimeError("get_saveables was called on an agent's worker object!")


class A2C(RLAgentBase):

    history_keys = ["actor_utility", "actor_utility_std", "actor_entropy", "actor_loss",
                    "values", "advantages", "critic_loss"]

    def __init__(self,
                 actor: keras.Model,
                 critic: keras.Model,
                 action_space,
                 absolute_memory_limit=10000,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None,
                 entropy_penalty_coef=0.,
                 copy_actor_models=False):

        super().__init__(action_space=action_space,
                         discount_factor_gamma=discount_factor_gamma,
                         state_preprocessor=state_preprocessor)

        self.actor_learner = actor
        self.critic_learner = critic
        self.num_actions = len(self.action_space)
        self.action_indices = np.arange(len(self.action_space))
        self.possible_actions_onehot = np.eye(len(self.action_space))
        self.entropy_penalty_coef = entropy_penalty_coef
        self._actor_train_function = self._make_actor_train_function()
        self.workers = []  # type: List[Actor]
        self.learning = False
        self.copy_actor_models = copy_actor_models
        self.absolute_memory_limit = absolute_memory_limit

    def set_learning_mode(self, switch: bool):
        pass

    def _update_memory_sampler(self):
        self.memory_sampler = ExperienceSampler([worker.memory for worker in self.workers])

    def create_worker(self, memory=None):
        if self.copy_actor_models:
            model = kerasic.copy_model(self.actor_learner)
        else:
            model = self.actor_learner
        if memory is None:
            memory = Experience(self.absolute_memory_limit)
        self.workers.append(
            Actor(model, self.action_space, memory, state_preprocessor=self.preprocess)
        )
        self._update_memory_sampler()
        return self.workers[-1]

    def dispatch_workers(self, n=1):
        if len(self.workers) > 0:
            raise RuntimeError("Workers already dispatched!")
        for i in range(n):
            self.create_worker(None)
        return self.workers

    def delete_workers(self):
        self.workers = []
        self.memory_sampler = None

    def _make_actor_train_function(self):
        K = keras.backend
        advantages = K.placeholder(shape=(None,))
        softmaxes = self.actor_learner.output
        actions = K.argmax(softmaxes, axis=-1)
        action_masks = K.one_hot(actions, num_classes=len(self.action_space))
        action_masks = K.stop_gradient(action_masks)

        probabilities = K.sum(action_masks * softmaxes, axis=1)
        log_prob = K.log(probabilities)
        entropy = -K.mean(log_prob)
        utilities = -log_prob * advantages
        utility = K.mean(utilities)
        utility_std = K.std(utilities)

        loss = -entropy * self.entropy_penalty_coef + utility
        updates = self.actor_learner.optimizer.get_updates(loss, self.actor_learner.weights)

        return K.function(inputs=[self.actor_learner.input, advantages],
                          outputs=[utility, utility_std, entropy, loss],
                          updates=updates)

    def sample(self, state, reward, done):
        if self.learning:
            raise RuntimeError("A2C's sample may not be called in learning mode!")

        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor_learner.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))

        return action

    def push_experience(self, state, reward, done):
        raise RuntimeError("A2C does not learn directly. Please create_workers()!")

    def _sync_workers(self):
        if not self.copy_actor_models:
            return
        W = self.actor_learner.get_weights()
        for worker in self.workers:
            worker.model.set_weights(W)

    def fit(self, batch_size=-1, reset_memory=True):
        S, S_, A, R, F = self.memory_sampler.sample(batch_size)
        assert len(S)

        S_ = self.preprocess(S_)
        S = self.preprocess(S)

        value_next = self.critic_learner.predict(S_)[..., 0]
        bellman_target = value_next * self.gamma + R
        bellman_target[F] = R[F]
        mean_bellman_error = self.critic_learner.train_on_batch(S, bellman_target)

        value = self.critic_learner.predict(S)[..., 0]
        advantage = bellman_target - value

        utility, std, entropy, loss = self._actor_train_function([S, advantage])

        if reset_memory:
            for worker in self.workers:
                worker.memory.reset()

        self._sync_workers()

        return {"actor_utility": utility,
                "actor_utility_std": std,
                "actor_entropy": entropy,
                "actor_loss": loss,
                "values": value.mean(),
                "advantages": advantage.mean(),
                "critic_loss": mean_bellman_error}

    def get_savables(self):
        return {"A2C_actor": self.actor_learner, "A2C_critic": self.critic_learner}
