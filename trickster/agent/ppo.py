from collections import defaultdict
from typing import List

import numpy as np
from keras import backend as K
from keras.models import Model

from ..experience import Experience, ExperienceSampler
from ..utility import numeric, kerasic
from ..abstract import RLAgentBase


class PPOWorker(RLAgentBase):

    def __init__(self,
                 actor: Model,
                 critic: Model,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 gae_lambda=0.95,
                 state_preprocessor=None):

        super().__init__(action_space,
                         memory,
                         discount_factor_gamma,
                         state_preprocessor)

        self.actor = actor
        self.critic = critic
        self.action_indices = np.arange(len(self.action_space))
        self.probabilities = []
        self.lmbda = gae_lambda

    def sample(self, state, reward, done):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        if self.learning:
            action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))
            self.probabilities.append(probabilities)
        else:
            action = np.squeeze(np.argmax(probabilities, axis=-1))

        self._push_step_to_direct_memory_if_learning(state, action, reward, done)

        return action

    def push_experience(self, state, reward, done):
        final_value = np.squeeze(self.critic.predict(state[None, ...]))
        states = np.array(self.states)
        next_states = np.array(self.states[1:] + [state])
        values = np.squeeze(self.critic.predict(states))
        values = np.append(values, final_value)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards[1:] + [reward])
        dones = np.array(self.dones[1:] + [done])
        probabilities = np.array(self.probabilities)

        returns = numeric.compute_gae(rewards, values[:-1], values[1:], dones, self.gamma, self.lmbda)

        memory = states, next_states, probabilities, actions, returns, values[1:], dones
        memory = self._filter_invalid_samples(*memory)

        lengths = set([len(array) for array in memory])
        assert len(lengths) == 1

        self.memory.remember(*memory)

        self._reset_direct_memory()
        self.probabilities = []


class PPO(RLAgentBase):

    history_keys = ("actor_loss", "actor_utility", "actor_utility_std", "actor_entropy", "actor_kld",
                    "values", "advantages", "critic_loss")

    def __init__(self,
                 actor,
                 critic,
                 action_space,
                 absolute_memory_limit: int=10000,
                 discount_factor_gamma=0.99,
                 gae_factor_lambda=0.95,
                 entropy_penalty_coef=0.005,
                 ratio_clip_epsilon=0.2,
                 target_kl_divergence=0.01,
                 state_preprocessor=None,
                 copy_models=False,
                 training_epochs=10):

        super().__init__(action_space,
                         memory=None,
                         discount_factor_gamma=discount_factor_gamma,
                         state_preprocessor=state_preprocessor)

        self.actor_learner = actor
        self.critic_learner = critic
        self.absolute_memory_limit = absolute_memory_limit
        self.entropy_penalty_coef = entropy_penalty_coef
        self.copy_models = copy_models
        self.lmbda = gae_factor_lambda
        self.ratio_clip = ratio_clip_epsilon
        self.target_kl = target_kl_divergence
        self.possible_actions_onehot = np.eye(len(self.action_space))
        self.action_indices = np.arange(len(self.action_space))
        self.training_epochs = training_epochs
        self.probabilities = []
        self._actor_train_fn = self._make_actor_train_function()
        self.workers = []  # type: List[PPOWorker]
        self.memory_sampler = None

    def create_worker(self, memory=None):
        if self.copy_models:
            actor = kerasic.copy_model(self.actor_learner)
            critic = kerasic.copy_model(self.critic_learner)
        else:
            actor = self.actor_learner
            critic = self.critic_learner
        if memory is None:
            memory = Experience(self.absolute_memory_limit)
        self.workers.append(
            PPOWorker(actor,
                      critic,
                      self.action_space,
                      memory,
                      self.gamma,
                      self.lmbda,
                      self.preprocess)
        )
        self.memory_sampler = ExperienceSampler([worker.memory for worker in self.workers])
        return self.workers[-1]

    def dispatch_workers(self, n=1):
        if len(self.workers) > 0:
            raise RuntimeError("Workers already created!")
        for i in range(n):
            self.create_worker(None)
        return self.workers

    def _make_actor_train_function(self):
        advantages = K.placeholder(shape=(None,), name="Advantage")
        action_onehot = K.placeholder(shape=(None, len(self.action_space)), name="Action_onehot")
        old_predictions = K.placeholder(shape=(None, len(self.action_space)), name="Old_predictions")

        advantages = advantages - K.mean(advantages)
        advantages = advantages / K.std(advantages)

        old_probabilities = K.sum(action_onehot * old_predictions, axis=1)
        old_log_prob = K.log(old_probabilities)

        old_log_prob = K.stop_gradient(old_log_prob)
        advantages = K.stop_gradient(advantages)

        new_pedictions = self.actor_learner.output
        new_probabilities = K.sum(action_onehot * new_pedictions, axis=1)
        new_log_prob = K.log(new_probabilities)
        entropy = -K.mean(new_log_prob)

        kld = K.mean(old_probabilities * (old_log_prob - new_log_prob))
        min_adv = K.switch(advantages > 0, (1+self.ratio_clip)*advantages, (1-self.ratio_clip)*advantages)
        ratio = K.exp(new_log_prob - old_log_prob)
        utilities = -K.minimum(ratio*advantages, min_adv)
        utility = K.mean(utilities)
        utility_stdev = K.std(utilities)

        loss = -entropy * self.entropy_penalty_coef + utility
        updates = self.actor_learner.optimizer.get_updates(loss, self.actor_learner.weights)

        return K.function(inputs=[self.actor_learner.input, old_predictions, advantages, action_onehot],
                          outputs=[utility, utility_stdev, entropy, kld, loss],
                          updates=updates)

    def sample(self, state, reward, done):
        if self.learning:
            raise RuntimeError("PPO's sample may not be called in learning mode!")

        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor_learner.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))

        return action

    def push_experience(self, state, reward, done):
        raise RuntimeError("PPO does not learn directly. Please create_workers()!")

    def _sync_workers(self):
        if not self.copy_models:
            return
        actor_W = self.actor_learner.get_weights()
        critic_W = self.critic_learner.get_weights()
        for worker in self.workers:
            worker.actor.set_weights(actor_W)
            worker.critic.set_weights(critic_W)

    def update_critic(self, batch_size):
        state, state_next, probability, action, returns, values, dones = self.memory_sampler.sample(batch_size)
        state = self.preprocess(state)
        critic_loss = self.critic_learner.train_on_batch(state, returns)
        return critic_loss

    def update_actor(self, batch_size):
        """ Keras pain

        return K.function(inputs=[self.actor_learner.input, old_predictions, advantages, action_onehot],
                  outputs=[utility, utility_stdev, entropy, approximate_kld, loss],
                  updates=updates)
        """
        state, state_next, probability, action, returns, values, dones = self.memory_sampler.sample(batch_size)
        state = self.preprocess(state)
        action_onehot = self.possible_actions_onehot[action]
        advantage = returns - values
        logs = list(self._actor_train_fn([state, probability, advantage, action_onehot]))
        return logs + [values.mean()] + [advantage.mean()]

    def update_step(self, batch_size, fit_actor=True, fit_critic=True, history=None):
        if history is None:
            history = defaultdict(list)
        if fit_critic:
            critic_loss = self.update_critic(batch_size)
            history["critic_loss"].append(critic_loss)
        if fit_actor:
            loss, loss_std, entropy, approximate_kld, combined_loss, value, advantage = self.update_actor(batch_size)
            history["actor_loss"].append(combined_loss)
            history["actor_utility"].append(loss)
            history["actor_utility_std"].append(loss_std)
            history["actor_entropy"].append(entropy)
            history["actor_kld"].append(approximate_kld)
            history["values"].append(value.mean())
            history["advantages"].append(advantage.mean())

    def fit(self, batch_size=32, fit_actor=True, fit_critic=True, reset_memory=True):
        history = defaultdict(list)
        if batch_size == -1:
            batch_size = self.memory_sampler.N

        updates_per_epoch = int(np.ceil(self.memory_sampler.N / batch_size))
        num_updates = self.training_epochs * updates_per_epoch
        for update in range(1, num_updates+1):
            self.update_step(batch_size, fit_actor, fit_critic, history)
            if history["actor_kld"][-1] > self.target_kl:
                break

        if reset_memory:
            for worker in self.workers:
                worker.memory.reset()

        history_new = {}
        for key in self.history_keys:
            history_new[key] = np.mean(history[key])

        return history_new

    def get_savables(self):
        return {"PPO_actor": self.actor_learner, "PPO_critic": self.critic_learner}
