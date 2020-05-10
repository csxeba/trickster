import tensorflow as tf
import gym


def gather_data(env: gym.Env, n: int, max_steps: int):
    rewards = []
    done = True
    step = 1
    for i in range(n):
        if max_steps is not None:
            done = done or step >= max_steps
        if done:
            step = 1
            env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        rewards.append(reward)
        step += 1
    return rewards


def set_weights(model: tf.keras.Model,
                env: gym.Env,
                num_steps: int = 100,
                set_bias: bool = True,
                set_weight: bool = True,
                env_max_steps: int = None):

    rewards = gather_data(env, num_steps, env_max_steps)

    output_layer: tf.keras.layers.Layer = model.layers[-1]
    while hasattr(output_layer, "layers"):
        output_layer = output_layer.layers[-1]
    W, b = output_layer.trainable_weights

    if set_bias:
        reward_mean = tf.reduce_mean(rewards)
        new_bias = tf.fill(b.shape, reward_mean)
        print(f" [Trickster] - Clever bias init: shape: {b.shape} value: {reward_mean}")
        b.assign(new_bias)

    if set_weight:
        reward_std = tf.math.reduce_std(rewards)
        orthogonal_initializer = tf.keras.initializers.Orthogonal(gain=reward_std)
        print(f" [Trickster] - Clever weight init: shape: {W.shape} gain: {reward_std}")
        new_weight = orthogonal_initializer(W.shape)
        W.assign(new_weight)

    env.reset()
