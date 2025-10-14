import tensorflow as tf
from tensorflow import keras

# CNN
inputs = keras.layers.Input(shape=[32,32,3])
x1 = keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(inputs)
x1 = keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x1)
prediction = keras.layers.MaxPool2D(pool_size=(2,2))(x1)
model = keras.models.Model(inputs=inputs,outputs=prediction,name="A.I LE")

model.summary()


"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

model_v1 = keras.Sequential([
    keras.layers.Dense(1024,input_dim=64),
    keras.layers.Activation("relu"),
    keras.layers.Dense(256),
    keras.layers.Activation("softmax")
])
model_v1.summary()
model_v1.inputs
model_v1.outputs
"""
