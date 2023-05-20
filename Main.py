from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

class_names = ["Lamborghini Aventador", "Porsche 911"]
Model = load_model("Model.h5")

while True:
    path = input("Path to img : ")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = np.array(img)
    img = img.reshape(1, 64, 64, 3)
    prediction = Model.predict(img)
    index = np.argmax(prediction)
    print(class_names[index])