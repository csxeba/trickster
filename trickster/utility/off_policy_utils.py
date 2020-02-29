from typing import Union

import gym
import tensorflow as tf

from ..model import policy, value


def sanitize_models_continuous(env: gym.Env,
                               actor: tf.keras.Model,
                               actor_target: Union[tf.keras.Model, None],
                               critic1: tf.keras.Model,
                               critic1_target: tf.keras.Model,
                               critic2: Union[tf.keras.Model, None],
                               critic2_target: Union[tf.keras.Model, None],
                               stochastic_actor: bool = False,
                               squash_actions: bool = False):

    actor_args = dict(env=env, stochastic=stochastic_actor, squash=squash_actions, wide=False,
                      sigma_mode=policy.SigmaMode.STATE_DEPENDENT)
    critic_args = dict(observation_space=env.observation_space, action_space=env.action_space, wide=True)

    if actor == "default":
        actor = policy.factory(**actor_args)
    if actor_target == "default":
        actor_target = policy.factory(**actor_args)

    if critic1 == "default":
        critic1 = value.QCritic(**critic_args)
    if critic1_target == "default":
        critic1_target = value.QCritic(**critic_args)

    if critic2 == "default":
        critic2 = value.QCritic(**critic_args)
    if critic2_target == "default":
        critic2_target = value.QCritic(**critic_args)

    return actor, actor_target, critic1, critic1_target, critic2, critic2_target


def sanitize_models_discreete(env: gym.Env,
                              model: tf.keras.Model,
                              target_network: tf.keras.Model,
                              use_target_network: bool = True):
    if model == "default":
        model = value.Q(env.observation_space, env.action_space)

    if use_target_network:
        if target_network == "default" or target_network is None:
            target_network = value.Q(env.observation_space, env.action_space)
    if not use_target_network:
        target_network = None
    return model, target_network
