# TF Keras
import tensorflow as tf
from tensorflow import keras

# First layer
input_shape = [4,30,30,3]
x1 = tf.random.normal(input_shape)
y1 = keras.layers.Conv2D(filters=32,
                        kernel_size=(3,3),
                        activation="relu",
                        input_shape=input_shape[1:])(x1)

# Tao CNN
inputs = keras.layers.Input(shape=[32,32,3])
x2 = keras.layers.Conv2D(filters=32,kernel_size=3,padding="same",activation="relu")(inputs)
x2 = keras.layers.Conv2D(filters=32,kernel_size=3,padding="same",activation="relu")(x2)
x2 = keras.layers.MaxPool2D()

#model = keras.models.Model(input=inputs,output=prediction)