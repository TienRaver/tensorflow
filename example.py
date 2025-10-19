import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

arr = np.random.randint(0,256,(500,500))
image = Image.fromarray(arr)
plt.figure(num=10,figsize=(8,6))
plt.imshow(image,cmap="gray")
plt.show()