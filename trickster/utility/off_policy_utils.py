from typing import Union

import gym
import tensorflow as tf

from ..model import arch


def sanitize_models_continuous(env: gym.Env,
                               actor: tf.keras.Model,
                               actor_target: Union[tf.keras.Model, None],
                               critic1: tf.keras.Model,
                               critic1_target: tf.keras.Model,
                               critic2: Union[tf.keras.Model, None],
                               critic2_target: Union[tf.keras.Model, None],
                               stochastic_actor: bool = False):

    action_maxima = env.action_space.high

    if actor == "default":
        actor = arch.Policy(env.observation_space, env.action_space,
                            stochastic=stochastic_actor,
                            squash_continuous=True, action_scaler=action_maxima, sigma_predicted=True)
    if actor_target == "default":
        actor_target = arch.Policy(env.observation_space, env.action_space,
                                   stochastic=stochastic_actor, squash_continuous=True, action_scaler=action_maxima)
    if critic1 == "default":
        critic1 = arch.QCritic(env.observation_space)
    if critic1_target == "default":
        critic1_target = arch.QCritic(env.observation_space)

    if critic2 == "default":
        critic2 = arch.QCritic(env.observation_space)
    if critic2_target == "default":
        critic2_target = arch.QCritic(env.observation_space)

    return actor, actor_target, critic1, critic1_target, critic2, critic2_target


def sanitize_models_discreete(env: gym.Env,
                              model: tf.keras.Model,
                              target_network: tf.keras.Model,
                              use_target_network: bool = True):
    if model == "default":
        model = arch.Q(env.observation_space, env.action_space)

    if use_target_network:
        if target_network == "default" or target_network is None:
            target_network = arch.Q(env.observation_space, env.action_space)
    if not use_target_network:
        target_network = None
    return model, target_network
