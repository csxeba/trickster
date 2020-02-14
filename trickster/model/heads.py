import gym
import tensorflow as tf
from tensorflow.keras import layers as tfl
import tensorflow_probability as tfp


class StochasticDiscreete(tf.keras.Model):

    """Stochastic discreete"""

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.logits = tfl.Dense(units=num_actions, activation="linear")
        self.num_outputs = num_actions

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=True, **kwargs):
        m = len(x)
        logits = self.logits(x)
        if training:
            distribution = tfp.distributions.Categorical(logits)
            action = distribution.sample(m)
            return action, distribution.prob(action)
        else:
            return tf.argmax(logits, axis=1)

    @tf.function(experimental_relax_shapes=True)
    def get_training_outputs(self, inputs, actions):
        distribution = self(inputs)
        entropy = distribution.entropy()
        log_prob = distribution.log_prob(actions)
        return log_prob, entropy


class DeterministicDiscreete(StochasticDiscreete):

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, *args, **kwargs):
        logits = self.logits(x)
        output = tf.argmax(logits)
        return output


class StochasticContinuous(tf.keras.Model):

    def __init__(self, num_actions: int, squash=None, sigma_predicted=True):
        super().__init__()
        self.mean_predictor = tfl.Dense(num_actions, activation="linear", name="actor_mean")
        self.sigma_predicted = sigma_predicted
        if self.sigma_predicted:
            self.log_stdev = tfl.Dense(num_actions, activation="linear", name="actor_log_stdev")
        else:
            self.log_stdev = tf.Variable(initial_value=tf.math.log(tf.ones([num_actions])), name="actor_log_stdev")
        self.squash = squash
        self.bijector = tfp.bijectors.Tanh()
        self.num_outputs = num_actions

    @tf.function(experimental_relax_shapes=True)
    def get_std(self, x):
        if self.sigma_predicted:
            std = tf.exp(self.log_stdev(x))
        else:
            std = tf.exp(self.log_stdev)[None, ...]
        return std

    @tf.function(experimental_relax_shapes=True)
    def get_training_outputs(self, inputs, actions):
        mean = self.mean_predictor(inputs)
        std = self.get_std(inputs)
        distribution = tfp.distributions.MultivariateNormalDiag(mean, std)
        entropy = distribution.entropy()
        if self.squash:
            distribution = self.bijector(distribution)
        log_prob = distribution.log_prob(actions)
        return log_prob, entropy

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True, **kwargs):
        mean = self.mean_predictor(inputs)
        if training:
            std = self.get_std(inputs)
            distribution = tfp.distributions.MultivariateNormalDiag(mean, std)
            if self.squash:
                distribution = self.bijector(distribution)
            action = distribution.sample()
            return action, distribution.prob(action)
        else:
            if self.squash:
                mean = tf.tanh(mean)
            return mean


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
    def call(self, inputs, training=True, **kwargs):
        action = self.action_predictor(inputs)
        if self.squash:
            action = tf.tanh(action)
        action = action * self.action_scaler
        return action


def factory(action_space: gym.spaces.Space,
            stochastic=True,
            squash_continuous=True,
            action_scaler=None,
            sigma_predicted=False):

    if stochastic:
        if isinstance(action_space, gym.spaces.Box):
            head = StochasticContinuous(action_space.shape[0],
                                        squash=squash_continuous,
                                        sigma_predicted=sigma_predicted)
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
