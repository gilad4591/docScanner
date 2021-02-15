# run - Scanner.py 1/page.jpg output.jpg
# Gilad Cohen  203722442
# Ofek Talker  311369961

import sys
import cv2
import imutils
import numpy as np

"""
PipeLine:
        readImage -> GrayScale -> Blurring  -> 
            Canny-Edge -> Find Contours -> Find Corners co-ordinates -> Crop the contour ->
                Sharping & Brightness correction (Binary threshold)
"""


#reshape the image
def orderPoints(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def read_image(path):
    img = cv2.imread(path)
    print("Image loaded")
    return img


# For terminal execute
path_img = sys.argv[1]
path_output = sys.argv[2]

# Read image in grayScale
image = read_image(path_img)
orig = imutils.resize(image, height=1000)
print("Work on a ratio of 1/1000")
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
print("Change image to grayscale")

# Blur image,remove noise and get canny edges
blurred = cv2.GaussianBlur(gray, (5, 5), 2)
edged = cv2.Canny(blurred, 30, 70, apertureSize=7)  # 30 = MinThreshold, 70 = MaxThreshold
print("Get canny edges of image")
# Extract contours of image
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)  # retrieve the contours as a list, with simple apprximation model
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print("Contour extracted")
# Extracts largest contours
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)
    # Extracts 4 points edges
    if len(approx) == 4:
        target = approx
        break

# find endpoints
approx = orderPoints(target)
print("Endpoints found")
pts = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
# bird eye view effect
op = cv2.getPerspectiveTransform(approx, pts)
dst = cv2.warpPerspective(gray, op, (image.shape[1], image.shape[0]))
print("Image cropped and changed to 'bird eye view effect'")
# Choose between adaptive threshold and binary.
# dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
ret1, dst = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
print("Scan effect on image")
#Save image to disk
cv2.imwrite(path_output, dst)
print("Finished succesfully")

