from typing import Union

import gym
import tensorflow as tf

from ..model import arch


def sanitize_models(env: gym.Env,
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
