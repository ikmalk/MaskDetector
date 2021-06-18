import numpy as np
from tensorflow.keras import layers, models
import pickle
import os

IMG_SIZE = 250
batch_size = 64

# X = training images, y = training labels

print("Loading data")
X = pickle.load(open("training_sets.pickle", "rb"))
y = pickle.load(open("training_label.pickle", "rb"))

# There are 255 values in each RGB
X = np.array(X / 255.0)
y = np.array(y)

"""
Creating training model and adding layers necessary
"""
print("Creating model")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

"""
Training start
"""
print("Training model")
model.fit(X, y, batch_size=batch_size, epochs=8, validation_split=0.2)

loss, accuracy = model.evaluate(X, y)
print(f"Loss={loss}")
print(f"Accuracy={accuracy}")

# Save model in file
model.save("MaskFaceClassifier.model")

# Color
# Loss = 0.0008296892043304993
# Accuracy =0.9997497200965881

"""
17553/17553 [==============================] - 185s 11ms/sample - loss: 0.0018 - accuracy: 0.9995
Loss=0.0017926976931164223
Accuracy=0.9995442628860474
"""