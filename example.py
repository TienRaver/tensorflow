import tensorflow as tf
import numpy as np

a = tf.constant([1,3,5,7,9],dtype=tf.float32)
b = tf.constant(3)

c = tf.math.multiply(a,b)
print(c)
print(c.shape())