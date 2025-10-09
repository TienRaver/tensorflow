import tensorflow as tf
import numpy as np

a = tf.random.normal([10,10],mean=2.5,stddev=0.25)
b = tf.random.normal([10,10],mean=1.25,stddev=0.22)
c = tf.math.add(a,b)

print(a,b,c)