from keras.models import Model, model_from_json


def copy_model(model: Model):
    arch = model.to_json()
    new_model = model_from_json(arch)
    new_model.set_weights(model.get_weights())
    return new_model


def meld_weights(target_model: Model, online_model: Model, mix_in_ratio: float):
    W = []
    mix_in_inverse = 1. - mix_in_ratio
    for old, new in zip(target_model.get_weights(), online_model.get_weights()):
        w = mix_in_inverse * old + mix_in_ratio * new
        W.append(w)
    target_model.set_weights(W)
    online_model.set_weights(W)
