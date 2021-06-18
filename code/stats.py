import plotly.graph_objects as go
import numpy as np
import os
import cv2
from tensorflow.keras import models
import random
import plotly.subplots as subplot

x = np.arange(10)

IMG_SIZE = 250
CATEGORIES = ["Mask", "NoMask"]
DATADIR = "../testdata"

testing_data = []
print("Loading model")
model = models.load_model("MaskFaceClassifier.model")

print("Reading sets")
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([new_array, class_num])
        except Exception as e:
            pass

random.shuffle(testing_data)

true = []
false = []

count = []
avg_percentage = []  # avg_percentage = total / n
total = 0
n = 0

print("Insert test data to model")
for img, label in testing_data:
    img2 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    prediction = model.predict(np.array([img2]) / 255)
    index = np.argmax(prediction)

    total += 100 * prediction[0][index]
    n += 1
    count.append(n)
    avg_percentage.append(total / n)

    if index == label:
        true.append("True")
    else:
        false.append("False")

fig = subplot.make_subplots(rows=1, cols=1)

fig.add_trace(go.Histogram(
    histfunc="count",
    x=true,
    name="Positive",
    marker_color='#88B9D5',
))

fig.add_trace(go.Histogram(
    histfunc="count",
    x=false,
    name="Negative",
    marker_color='#FF69B4',
)
)

fig.show()


fig2 = go.Figure(data=go.Scatter(x=count, y=avg_percentage))
fig2.update_xaxes(title_text="Number of images")
fig2.update_yaxes(title_text="Mean of confidence percentage")
fig2.show()
