import cv2
import sys
import logging as log
import datetime as dt
import numpy as np
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

faceOverlay = cv2.imread('dogeface.png', -1)
o_height = faceOverlay.shape[0]
o_width = faceOverlay.shape[1]

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        x_offset=x
        y_offset=y

        # resize overlay image to fit face
        down_width = w
        down_height = h
        scale_down_x = down_width/o_width
        scale_down_y = down_height/o_height
        scaled_f_down = cv2.resize(faceOverlay, None, fx= scale_down_x, fy= scale_down_y, interpolation= cv2.INTER_LINEAR)

        y1, y2 = y_offset, y_offset + scaled_f_down.shape[0]
        x1, x2 = x_offset, x_offset + scaled_f_down.shape[1]
        alpha_s = scaled_f_down[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s

        # overlay small image over face w/ alpha blending
        for c in range(0, 3):
            try:
                frame[y1:y2, x1:x2, c] = (alpha_s * scaled_f_down[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
