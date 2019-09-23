import keras

from ..utility import kerasic


def _conv(x, width, ds=True, activate=True, batch_norm=True):
    strides = 2 if ds else 1
    x = keras.layers.Conv2D(width, (5, 5), strides=strides, padding="same")(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    if activate:
        x = keras.layers.LeakyReLU()(x)
    return x


def _dense(flat_x, width, activate=True, batch_norm=True):
    x = keras.layers.Dense(width, activation="linear")(flat_x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    if activate:
        x = keras.layers.LeakyReLU()(x)
    return x


def _conv_features(x, batch_norm=True):
    x = _conv(x, 16, ds=True, activate=True, batch_norm=batch_norm)
    x = _conv(x, 32, ds=True, activate=True, batch_norm=batch_norm)
    x = _conv(x, 32, activate=True, batch_norm=batch_norm)

    return x


def _narrow_mlp_head(flat_x, num_outputs, activation="linear", batch_norm=True):
    x = _dense(flat_x, 64, activate=True, batch_norm=batch_norm)
    x = _dense(x, 64, activate=True, batch_norm=batch_norm)
    x = _dense(x, num_outputs, activate=False, batch_norm=False)
    x = keras.layers.Activation(activation)(x)
    return x


def _cnn(input_shape, output_dim, output_activation, batch_norm=True, as_model=True):
    inputs = keras.Input(input_shape)
    features = _conv_features(inputs, batch_norm=batch_norm)
    flat_features = keras.layers.GlobalAveragePooling2D()(features)
    outputs = _narrow_mlp_head(flat_features, output_dim, activation=output_activation, batch_norm=batch_norm)
    if as_model:
        model = keras.Model(inputs, outputs)
        return model
    else:
        return inputs, outputs


def cnn_policy_categorical(input_shape, output_dim, adam_lr=1e-4, batch_norm=False):
    model = _cnn(input_shape, output_dim, output_activation="softmax", batch_norm=batch_norm)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="categorical_crossentropy")
    return model


def cnn_policy_continuous(input_shape, output_dim,
                          output_activation="linear",
                          clip_range=None,
                          adam_lr=1e-4,
                          batch_norm=False):

    inputs, outputs = _cnn(input_shape, output_dim, output_activation=output_activation, batch_norm=batch_norm,
                           as_model=False)
    if clip_range:
        if isinstance(clip_range, int):
            clip_range = -clip_range, clip_range
        outputs = kerasic.clip(outputs, *clip_range)
    model = keras.Model(inputs, outputs)
    model.compile(keras.optimizers.Adam(adam_lr), loss="mse")
    return model


def cnn_critic(input_shape, output_dim, adam_lr=1e-4, batch_norm=False):
    model = _cnn(input_shape, output_dim, output_activation="linear", batch_norm=batch_norm)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="categorical_crossentropy")
    return model


def _cnn_ddpg_critic(input_shape, action_dim, adam_lr=1e-4, batch_norm=False):
    observation_inputs = keras.Input(input_shape)
    action_inputs = keras.Input([action_dim])
    observation_features = _conv_features(observation_inputs, batch_norm)
    action_features = _dense(action_inputs, 32, activate=True, batch_norm=True)
    x = keras.layers.concatenate([observation_features, action_features])
    q = _narrow_mlp_head(x, num_outputs=1, activation="linear", batch_norm=batch_norm)
    critic = keras.Model([observation_inputs, action_inputs], q)
    critic.compile(keras.optimizers.Adam(adam_lr), loss="mse")
    return critic


def cnn_pg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4, batch_norm=False):
    actor = cnn_policy_categorical(input_shape, output_dim, actor_lr, batch_norm)
    critic = cnn_critic(input_shape, output_dim=1, adam_lr=critic_lr, batch_norm=batch_norm)
    return actor, critic


def cnn_ddpg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4, batch_norm=False, clip_range=None,
                          num_critics=1):

    actor = cnn_policy_continuous(input_shape, output_dim, clip_range=clip_range, adam_lr=actor_lr,
                                  batch_norm=batch_norm)
    critics = []
    critic = None
    for i in range(1, num_critics+1):
        critic = _cnn_ddpg_critic(input_shape, output_dim, critic_lr, batch_norm)
        critics.append(critic)
    if num_critics > 1:
        return actor, critics
    return actor, critic
