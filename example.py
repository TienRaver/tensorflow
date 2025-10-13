import tensorflow as tf
from tensorflow import keras

# CNN
inputs = keras.layers.Input(shape=[32,32,3])
x1 = keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(inputs) # L1
x1 = keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x1) # L2
prediction = keras.layers.MaxPool2D(pool_size=(2,2))(x1) # Downsize
model = keras.models.Model(inputs=inputs,outputs=prediction,name="A.I LE")

model.summary()
