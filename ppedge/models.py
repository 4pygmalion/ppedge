import tensorflow as tf


def build_model(model: str = "vgg", n_class: int = 10) -> tf.keras.Model:
    """Build keras defualt model for test

    Args:
        model (str): tensorflow supported model name
        n_class (int): number of classes to classify

    Return:
        model (tf.keras.Model)
    """

    if model == "vgg":
        base_model = tf.keras.applications.VGG19(
            include_top=False, input_shape=(256, 256, 3), classes=n_class
        )
        base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10, activation="sigmoid")(x)
    model = tf.keras.models.Model(base_model.input, x)

    return model


def inhance_graph_privacy(model: str = "vgg") -> tf.keras.Model:
    # ADD Batch normalization
    base_model = tf.keras.models.Sequential()
    for layer in model.layers:

        if "conv1" in layer.name:
            base_model.add(layer)
            base_model.add(
                tf.keras.layers.MaxPool2D(
                    pool_size=(2, 2), strides=(1, 1), padding="valid"
                )
            )
            base_model.add(tf.keras.layers.BatchNormalization())
        elif "conv2" in layer.name:
            base_model.add(layer)
            base_model.add(tf.keras.layers.BatchNormalization())
        else:
            base_model.add(layer)

    x = base_model.output
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10, activation="sigmoid")(x)
    base_model = tf.keras.models.Model(base_model.input, x)

    return base_model
