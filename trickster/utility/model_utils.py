import tensorflow as tf


def copy_model(model: tf.keras.Model, rename_to=None, copy_weights=True):
    arch = model.to_json()
    if rename_to is not None:
        import json
        arch_json = json.loads(arch)
        arch_json["config"]["name"] = rename_to
        arch = json.dumps(arch_json)
    new_model = tf.keras.models.model_from_json(arch)
    if copy_weights:
        new_model.set_weights(model.get_weights())
    return new_model


def meld_weights(target_model: tf.keras.Model, online_model: tf.keras.Model, mix_in_ratio: float):
    W = []
    mix_in_inverse = 1. - mix_in_ratio
    d = 0.
    for old, new in zip(target_model.get_weights(), online_model.get_weights()):
        w = mix_in_inverse * old + mix_in_ratio * new
        W.append(w)
        d += ((w - new) ** 2.).sum()
    target_model.set_weights(W)
    return d


def clip(x, min_value, max_value):
    return keras.layers.Lambda(lambda xx: keras.backend.clip(xx, min_value, max_value))(x)
