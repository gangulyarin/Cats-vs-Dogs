from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K

class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = models.Sequential()
        inputShape = (height,width,depth)
        #if K.image_data_format()=="channels_first":
        #    inputShape = (depth,height,width)
        model.add(layers.Conv2D(32,(3,3),padding="same",input_shape=inputShape))
        model.add(layers.Activation("relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(classes))
        model.add(layers.Softmax())

        return model
