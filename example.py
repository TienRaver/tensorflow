
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Xay dung model
model = keras.models.Model(
    keras.layers.Input(shape=[])
)

"""
# Tạo dữ liệu giả: 2000 mẫu, mỗi mẫu có 64 đặc trưng
data = np.random.random((2000,64))
labels = np.random.randint(0,10,(2000,))
labels = tf.keras.utils.to_categorical(labels, num_classes=10)

valid_data = np.random.random((500,64))
valid_labels = tf.keras.utils.to_categorical(np.random.randint(0,10,(500,)), num_classes=10)

test_data = np.random.random((500,64))
test_labels = tf.keras.utils.to_categorical(np.random.randint(0,10,(500,)), num_classes=10)

# Xây dựng model
model_v1 = keras.Sequential([
    keras.Input(shape=(64,)),
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model_v1.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Huấn luyện
model_v1.fit(data, labels, epochs=10, batch_size=50, validation_data=(valid_data, valid_labels))

# Đánh giá
model_v1.evaluate(test_data, test_labels)
"""
