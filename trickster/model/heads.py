import gym
import tensorflow as tf
from tensorflow.keras import layers as tfl
import tensorflow_probability as tfp


class StochasticDiscreete(tf.keras.Model):

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.logits = tfl.Dense(units=num_actions, activation="linear")
        self.num_outputs = num_actions

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, *args, **kwargs):
        logits = self.logits(x)
        output = tfp.distributions.Categorical(logits)
        return output


class DeterministicDiscreete(StochasticDiscreete):

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, *args, **kwargs):
        logits = self.logits(x)
        output = tf.argmax(logits)
        return output


class StochasticContinuous(tf.keras.Model):

    def __init__(self, num_actions: int, squash=True, stochastic=True):
        super().__init__()
        self.mean_predictor = tfl.Dense(num_actions, activation="linear")
        self.log_stdev = tf.Variable(initial_value=tf.math.log(tf.ones([num_actions])), name="actor_log_stdev")
        self.squash = squash
        self.stochastic = stochastic
        if stochastic:
            self.bijector = tfp.bijectors.Tanh()
        self.num_outputs = num_actions

    def _parallelizable_part(self, inputs):
        batch_size = len(inputs)
        mean = self.mean_predictor(inputs)
        stdev = tf.stack([tf.exp(self.log_stdev)] * batch_size, axis=0)
        return mean, stdev

    def call(self, inputs, *args, **kwargs):
        mean, stdev = self._parallelizable_part(inputs)
        output = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stdev)
        if self.squash:
            output = self.bijector(output)
        return output


class DeterministicContinuous(tf.keras.Model):

    def __init__(self, num_actions: int, squash=True, action_scaler=None):
        super().__init__()
        self.num_actions = num_actions
        self.action_predictor = tfl.Dense(num_actions, activation="linear")
        self.squash = squash
        self.num_outputs = num_actions
        if action_scaler is None:
            action_scaler = 1.
        self.action_scaler = action_scaler

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, *args, **kwargs):
        action = self.action_predictor(inputs)
        if self.squash:
            action = tf.tanh(action)
        action = action * self.action_scaler
        return action


def factory(action_space: gym.spaces.Space,
            stochastic=True,
            squash_continuous=True,
            action_scaler=None):

    if stochastic:
        if isinstance(action_space, gym.spaces.Box):
            head = StochasticContinuous(action_space.shape[0], squash=squash_continuous)
        elif isinstance(action_space, gym.spaces.Discrete):
            head = StochasticDiscreete(action_space.n)
        else:
            raise RuntimeError(f"Weird action space type: {type(action_space)}")

    else:
        if isinstance(action_space, gym.spaces.Box):
            head = DeterministicContinuous(action_space.shape[0], squash_continuous, action_scaler)
        else:
            raise RuntimeError(f"Weird action space type for deterministic head: {type(action_space)}")
    return head
