import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import copy
import numpy as np
import os
from tensorflow.keras import models

"""
Video capture face detector courtesy of 
shantu https://github.com/shantnu/Webcam-Face-Detect 
"""

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

IMG_SIZE = 250
CATEGORIES = ["Mask", "NoMask"]

print("Loading tensorflow")
model = models.load_model("MaskFaceClassifier.model")

index = 0
n = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    n += 1
    if n == 3:
        n = 0

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    img = copy.deepcopy(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    font = cv2.FONT_HERSHEY_DUPLEX

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        try:
            if n == 0:
                img2 = img[y - 50:y + h + 50, x - 50:x + w + 50]
                img3 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))

                prediction = model.predict(np.array([img3]) / 255)
                index = np.argmax(prediction)
                name = ""
                color = (0, 0, 0)

            if index == 0:
                name = "Correct Mask"
                color = (0, 255, 0)
            elif index == 1:
                name = "Incorrect Mask"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 20), font, 0.7, color, 1)
        except Exception as e:
            pass

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()