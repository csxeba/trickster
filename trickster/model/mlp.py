import keras


K = keras.backend


def _wide_mlp_layers(input_shape, output_dim, activation="linear"):
    return [keras.layers.Dense(400, kernel_initializer="he_normal", input_shape=input_shape),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(300, kernel_initializer="he_normal"),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(output_dim, activation=activation)]


def _wide_ddgp_critic(input_shape, output_dim, adam_lr):
    state_input = keras.Input(input_shape)
    action_input = keras.Input([output_dim])
    x = keras.layers.concatenate([state_input, action_input])
    x = keras.layers.Dense(400, kernel_initializer="he_normal")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.LeakyReLU()(x)
    q = keras.layers.Dense(1, activation="linear")(x)
    critic = keras.Model([state_input, action_input], q)
    critic.compile(keras.optimizers.Adam(adam_lr), loss="mse")
    return critic


def wide_mlp_actor_categorical(input_shape, output_dim, adam_lr=1e-3):
    model = keras.Sequential(_wide_mlp_layers(input_shape, output_dim) + [keras.layers.Softmax()])
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="categorical_crossentropy")
    return model


def wide_mlp_actor_continuous(input_shape, output_dim, adam_lr=1e-3, activation="linear", action_range=None):
    additional_layers = []
    if not isinstance(action_range, tuple):
        action_range = (-action_range, action_range)
    if action_range is not None:
        additional_layers.append(keras.layers.Lambda(lambda x: K.clip(x, action_range[0], action_range[1])))
    model = keras.Sequential(_wide_mlp_layers(input_shape, output_dim, activation) + additional_layers)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model


def wide_mlp_critic_network(input_shape, output_dim, adam_lr=1e-3):
    model = keras.Sequential(_wide_mlp_layers(input_shape, output_dim))
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model


def wide_pg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4):
    actor = wide_mlp_actor_categorical(input_shape, output_dim, actor_lr)
    critic = wide_mlp_critic_network(input_shape, 1, critic_lr)
    return actor, critic


def wide_ddpg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4,
                           actor_activation="linear", action_range=None, num_critics=1):

    actor = wide_mlp_actor_continuous(input_shape, output_dim, actor_lr, actor_activation, action_range)
    critics = []
    critic = None
    for i in range(num_critics):
        critic = _wide_ddgp_critic(input_shape, output_dim, critic_lr)
        critics.append(critic)
    if num_critics > 1:
        return actor, critics
    return actor, critic


def wide_dueling_q_network(input_shape, output_dim, adam_lr=1e-3):
    inputs = keras.Input(input_shape)
    h1 = keras.layers.Dense(400)(inputs)
    h1 = keras.layers.LeakyReLU()(h1)
    h2 = keras.layers.Dense(300)(h1)
    h2 = keras.layers.LeakyReLU()(h2)

    value = keras.layers.Dense(1, activation="linear")(h2)
    advantage = keras.layers.Dense(output_dim, activation="linear")(h2)

    q = keras.layers.add([value, advantage])

    model = keras.Model(inputs=inputs, outputs=q)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model
