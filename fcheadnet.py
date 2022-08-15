import tensorflow as tf

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        headModel = baseModel.output
        headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        headModel = tf.keras.layers.Dense(D, activation="relu")(headModel)
        headModel = tf.keras.layers.Dropout(0.5)(headModel)
        headModel = tf.keras.layers.Dense(classes,activation="softmax")(headModel)

        return headModel