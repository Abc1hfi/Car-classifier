from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras.models
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

Y = []
Carpics = []


for i in range(5):
    print(f"Porsche00{i} added")
    img = cv2.imread(f"Dataset/Porsche/Porsche 911/00{i}.jpeg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))

    Carpics.append(img)
    Y.append(1)

for i in range(5):
    print(f"Lamborghini00{i} added")
    img = cv2.imread(f"Dataset/Lamborghini/Aventador/00{i}.jpeg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))

    Carpics.append(img)
    Y.append(0)

Carpics = np.array(Carpics, dtype=np.uint8)
Y = np.array(Y)  # Convert labels to numpy array

X_train, X_test, Y_train, Y_test = train_test_split(Carpics, Y, test_size=0.2, random_state=45)

# Reshape the data
#X_train = X_train.reshape((-1, 64, 64, 3))
#X_test = X_test.reshape((-1, 64, 64, 3))

class_names = ["Lamborghini Aventador", "Porsche 911"]

model = keras.models.Sequential([
    Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (5, 5), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(128, (3, 3), activation="relu"),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(20)
])

model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])



hist = model.fit(X_train, Y_train, epochs=50, verbose=1, validation_data=(X_test, Y_test))

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and validation accuracy")
plt.legend()

plt.show()

model.save("Model.h5")