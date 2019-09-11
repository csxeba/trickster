import keras


def _wide_mlp_layers(input_shape, output_dim):
    return [keras.layers.Dense(400, activation="relu", input_shape=input_shape),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(output_dim, activation="linear")]


def wide_mlp_actor_categorical(input_shape, output_dim, adam_lr=1e-3):
    model = keras.Sequential(_wide_mlp_layers(input_shape, output_dim) + [keras.layers.Softmax()])
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="categorical_crossentropy")
    return model


def wide_mlp_actor_continuous(input_shape, output_dim, adam_lr=1e-3):
    model = keras.Sequential(_wide_mlp_layers(input_shape, output_dim))
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


def wide_ddpg_actor_critic(input_shape, output_dim, actor_lr=1e-4, critic_lr=1e-4):
    actor = wide_mlp_actor_continuous(input_shape, output_dim, actor_lr)
    critic = wide_mlp_critic_network(input_shape, output_dim, critic_lr)
    return actor, critic


def wide_dueling_q_network(input_shape, output_dim, adam_lr=1e-3):
    inputs = keras.Input(input_shape)
    h1 = keras.layers.Dense(400, activation="relu")(inputs)
    h2 = keras.layers.Dense(300, activation="relu")(h1)

    value = keras.layers.Dense(1, activation="linear")(h2)
    advantage = keras.layers.Dense(output_dim, activation="linear")(h2)

    q = keras.layers.add([value, advantage])

    model = keras.Model(inputs=inputs, outputs=q)
    model.compile(optimizer=keras.optimizers.Adam(adam_lr), loss="mse")
    return model
