# Thuat toan lan truyen nguoc backpropagation

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Tao x_true, y_true
np.random.seed(0)
x_vals = np.random.normal(1,0.1,100).astype(np.float32)
y_vals = (x_vals*np.random.normal(1,0.05,100)-0.15).astype(np.float32) # y = w*x-1

# Tao Neuron de tinh y_pred
def my_output(x,weights,biases):
    return tf.add(tf.multiply(x,weights),biases)

# Tao Loss
def my_loss(y_true,y_pred): # loss binh phuong trung binh
    return tf.reduce_mean(tf.square(y_pred-y_true))

# Tao Optimizer
my_optimizer = tf.optimizers.SGD(learning_rate=0.02)

# Tao Weight, bias
tf.random.set_seed(1)
np.random.seed(0)
weights = tf.Variable(tf.random.normal(shape=[1]))
biases = tf.Variable(tf.random.normal(shape=[1]))
history = list()

# Thuat toan backpropagation
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = x_vals[rand_index] # Input x_true
    rand_y = y_vals[rand_index] # Output y_true
    with tf.GradientTape() as tape:
        y_pred = my_output(rand_x,weights,biases) # Tinh y_pred
        loss = my_loss(rand_y,y_pred) # Tinh loss
    history.append(loss.numpy())
    gradients = tape.gradient(loss,[weights,biases]) # Tinh dao ham (gradient) Loss cho weights, biass
    my_optimizer.apply_gradients(zip(gradients,[weights,biases])) # Cap nhat weights, bias theo gradient vua tinh duoc
    if (i+1)%25 == 0:
        print(f"Step {i+1} Weight: {weights.numpy()} Biases : {biases.numpy()}")
        print(f"Loss = {loss.numpy()}")

# In ra ham Loss
plt.plot(history)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()