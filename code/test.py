import cv2
import numpy as np
import os
from tensorflow.keras import models


CATEGORIES = ["Mask", "NoMask"]
IMG_SIZE = 150

model = models.load_model("MaskFaceClassifier.model")

path = "../testdata"

index = []

for img in os.listdir(path):
    image = cv2.imread(os.path.join(path, img), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    prediction = model.predict(np.array([image]) / 255)
    i = np.argmax(prediction)
    index.append(i)


for n in index:
    print(f"{CATEGORIES[n]}")
