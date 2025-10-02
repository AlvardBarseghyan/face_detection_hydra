import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import VGG16

def build_fc(config, input_shape=(224, 224, 3)):
    model = Sequential([layers.Input(shape=input_shape), layers.Flatten()])
    for units, dropout in zip(config.hidden_units, config.dropouts):
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(config.num_classes, activation="softmax"))
    return model

def build_cnn(config, input_shape=(224, 224, 3)):
    model = Sequential([layers.Input(shape=input_shape)])
    for f in config.filters:
        model.add(layers.Conv2D(f, (config.kernel_size, config.kernel_size), activation="relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(config.num_classes, activation="softmax"))
    return model

def build_vgg(config, input_shape=(224, 224, 3)):
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg.trainable = config.trainable
    model = Sequential([
        vgg,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(config.num_classes, activation="softmax")
    ])
    return model

def get_model(config):
    if config.name == "fc":
        return build_fc(config)
    elif config.name == "cnn":
        return build_cnn(config)
    elif config.name == "vgg":
        return build_vgg(config)
    else:
        raise ValueError(f"Unknown model: {config.name}")

