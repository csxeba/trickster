from keras.models import Model, model_from_json


def copy_model(model: Model):
    arch = model.to_json()
    new_model = model_from_json(arch)
    new_model.set_weights(model.get_weights())
    return new_model
