import tensorflow as tf

# # TODO: 마지막 layer name도 반환
# def build_model(model="vgg"):

#     if model == "vgg":
#         vgg19 = tf.keras.applications.VGG19(
#             include_top=False,
#             input_shape=(256, 256, 3),
#             classes=10
#         )
#         vgg19.trainable = False

#     # ADD Batch normalization
#     base_model = tf.keras.models.Sequential()
#     for layer in vgg19.layers:

#         if "conv1" in layer.name:
#             base_model.add(layer)
#             base_model.add(
#                 tf.keras.layers.MaxPool2D(
#                     pool_size=(2, 2), strides=(1, 1), padding="valid"
#                 )
#             )
#             base_model.add(tf.keras.layers.BatchNormalization())
#         elif "conv2" in layer.name:
#             base_model.add(layer)
#             base_model.add(tf.keras.layers.BatchNormalization())
#         else:
#             base_model.add(layer)

#     x = base_model.output
#     x = tf.keras.layers.Dense(1000, activation="relu")(x)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dense(10, activation="sigmoid")(x)
#     model = tf.keras.models.Model(base_model.input, x)

#     return model
