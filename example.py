# Cach 1: Tao layer truc tiep trong Sequential
import tensorflow as tf
from tensorflow import keras
import numpy as np

model_v1 = keras.Sequential([
    keras.layers.Dense(1024,input_dim=64),
    keras.layers.Activation("relu"),
    keras.layers.Dense(256),
    keras.layers.Activation("softmax")])
model_v1.summary()

# Cach 2: Tao Sequential roi gan tung layer vao
model_v2 = keras.Sequential()
model_v2.add(keras.layers.Dense(1024,input_dim=64))
model_v2.add(keras.layers.Activation("relu"))
model_v2.add(keras.layers.Dense(256))
model_v2.add(keras.layers.Activation("softmax"))
model_v2.summary()