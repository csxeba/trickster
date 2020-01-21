"""
This module contains some basic MLP architectures, which are more-or-less standard for
Deep Reinforcement Learning.
"""

from tensorflow import keras

K = keras.backend


def _dense(x, units, activation, batch_norm):
    activation_layer = {"leakyrelu": keras.layers.LeakyReLU}.get(
        activation,
        lambda: keras.layers.Activation(activation))
    x = keras.layers.Dense(units)(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = activation_layer()(x)
    return x


def _wide_mlp_layers(inputs, output_dim, output_activation, batch_norm):
    x = _dense(inputs, 400, "leakyrelu", batch_norm)
    x = _dense(x, 300, "leakyrelu", batch_norm)
    x = _dense(x, output_dim, output_activation, batch_norm=False)
    return x


def _wide_ddgp_critic(input_shape, output_dim, adam_lr, batch_norm):
    state_input = keras.Input(input_shape)
    action_input = keras.Input([output_dim])
    x = keras.layers.concatenate([state_input, action_input])
    q = _wide_mlp_layers(inputs=x, output_dim=1, output_activation="linear", batch_norm=batch_norm)
    critic = keras.Model([state_input, action_input], q)
    critic.compile(keras.optimizers.Adam(adam_lr), loss="mse")
    return critic


def _wide_categorical_action_critic(input_shape, action_dim):
    ...


def wide_mlp_actor_categorical(input_shape, output_dim, adam_lr=1e-3, batch_norm=False):
    state_input = keras.Input(input_shape)
    actions = _wide_mlp_layers(state_input, output_dim, output_activation="linear", batch_norm=batch_norm)
    actor = keras.Model(state_input, actions)
    actor.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="categorical_crossentropy")
    return actor


def wide_mlp_actor_continuous(input_shape, output_dim, adam_lr=1e-3, activation="linear", action_range=None,
                              batch_norm=False):
    if not isinstance(action_range, tuple):
        action_range = (-action_range, action_range)
    state_input = keras.Input(input_shape)
    action = _wide_mlp_layers(state_input, output_dim, output_activation=activation, batch_norm=batch_norm)
    if action_range is not None:
        action = keras.layers.Lambda(lambda x: K.clip(x, action_range[0], action_range[1]))(action)
    model = keras.Model(state_input, action)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model


def wide_mlp_critic(input_shape, output_dim, adam_lr=1e-3, batch_norm=False):
    state_input = keras.Input(input_shape)
    output = _wide_mlp_layers(state_input, output_dim, output_activation="linear", batch_norm=batch_norm)
    model = keras.Model(state_input, output)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model


def wide_pg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4, batch_norm=False):
    actor = wide_mlp_actor_categorical(input_shape, output_dim, actor_lr, batch_norm)
    critic = wide_mlp_critic(input_shape, 1, critic_lr, batch_norm)
    return actor, critic


def wide_ddpg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4,
                           actor_activation="linear", action_range=None, num_critics=1, batch_norm=False):

    actor = wide_mlp_actor_continuous(input_shape, output_dim, actor_lr, actor_activation, action_range, batch_norm)
    critics = []
    critic = None
    for i in range(1, num_critics+1):
        critic = _wide_ddgp_critic(input_shape, output_dim, critic_lr, batch_norm)
        critics.append(critic)
    if num_critics > 1:
        return actor, critics
    return actor, critic


def wide_dueling_q_network(input_shape, output_dim, adam_lr=1e-3, batch_norm=False):
    inputs = keras.Input(input_shape)
    h1 = _dense(inputs, 400, "leakyrelu", batch_norm)
    h2 = _dense(h1, 300, "leakyrelu", batch_norm)

    value = _dense(h2, 1, activation="linear", batch_norm=False)
    advantage = _dense(h2, output_dim, activation="linear", batch_norm=False)

    q = keras.layers.add([value, advantage])

    model = keras.Model(inputs=inputs, outputs=q)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model
