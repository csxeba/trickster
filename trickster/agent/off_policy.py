import tensorflow as tf

from .abstract import RLAgentBase
from ..utility import model_utils


class OffPolicy(RLAgentBase):

    transition_memory_keys = ["state", "state_next", "action", "reward", "done"]

    def __init__(self,
                 actor: tf.keras.Model = None,
                 actor_target: tf.keras.Model = None,
                 critic: tf.keras.Model = None,
                 critic_target: tf.keras.Model = None,
                 critic2: tf.keras.Model = None,
                 critic2_target: tf.keras.Model = None,
                 memory_buffer_size: int = 10000,
                 discount_gamma: float = 0.99,
                 polyak_tau: float = 0.01):

        super().__init__(memory_buffer_size, separate_training_memory=False)

        self.actor = actor
        self.critic = critic
        self.critic2 = critic2

        self.has_actor_target = True
        self.has_critic_target = True
        self.has_critic2_target = True

        if actor_target is None:
            actor_target = actor
            self.has_actor_target = False
        if critic_target is None:
            critic_target = critic
            self.has_critic_target = False
        if critic2_target is None:
            critic2_target = critic2
            self.has_critic2_target = False

        self.actor_target = actor_target
        self.critic_target = critic_target
        self.critic2_target = critic2_target

        self.gamma = discount_gamma
        self.tau = polyak_tau

    def _set_transition(self, state, action, reward, done):
        if self.timestep > 0:
            self.transition.set(state_next=state, reward=reward, done=done)
            assert self.transition.ready
            self.transition_memory.store(self.transition)

        self.timestep += 1
        if done:
            self.timestep = 0
            self.episodes += 1
        else:
            self.transition.set(state=state, action=action)

    def sample(self, state, reward, done):
        raise NotImplementedError

    def end_trajectory(self):
        pass

    def get_savables(self) -> dict:
        result = {}
        for model_name in ["actor", "actor_target", "critic", "critic_target", "critic2", "critic2_target"]:
            model = getattr(self, model_name)
            if model is not None:
                result[self.__class__.__name__ + "_" + model_name] = model
        return result

    def update_targets(self):
        if self.has_actor_target:
            model_utils.meld_weights(self.actor_target, self.actor, self.tau)
        if self.has_critic_target:
            model_utils.meld_weights(self.critic_target, self.critic, self.tau)
        if self.has_critic2_target:
            model_utils.meld_weights(self.critic2_target, self.critic2, self.tau)

    def fit(self, batch_size=None):
        raise NotImplementedError
