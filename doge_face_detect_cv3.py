import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
faceOverlay = cv2.imread('dogeface.png', -1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

o_height = faceOverlay.shape[0]
o_width = faceOverlay.shape[1]

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
            image[y1:y2, x1:x2, c] = (alpha_s * scaled_f_down[:, :, c] + alpha_l * image[y1:y2, x1:x2, c])
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")

cv2.imshow("Faces found", image)
cv2.waitKey(0)
