import numpy as np
import cv2
import os
import random
import pickle

"""
Before declaring the datasets path,
the datasets must be seperated in different folder,
folder in this case acts as label 
Example:

Datasets Folder
    |
    |- Label 1
        |---Images 1
        |---Images 2
    |- Label 2
        |---Images 1
        |---Images 2
"""
DATADIR = "../datasets"

# Names of folders in datasets folder
CATEGORIES = ["Mask", "NoMask"]

# Resize the image in file
IMG_SIZE = 250

training_data = []

print("Producing pickle")

"""
Read all images in datasets based on its path and label
"""
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

# Shuffling so datasets is not sorted to make sure
# the training went smoothly
random.shuffle(training_data)

"""
training sets is images data
training label is images label in CATEGORIES
"""
training_sets = []
training_label = []

for features, label in training_data:
    training_sets.append(features)
    training_label.append(label)

X = np.array(training_sets).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

"""
Store training sets and label in pickle library
so no need to read image repetitively
"""

pickle_out = open("training_sets.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("training_label.pickle", "wb")
pickle.dump(training_label, pickle_out)
pickle_out.close()
print("Done producing pickle")