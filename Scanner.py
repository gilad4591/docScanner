# run - Scanner.py 1/page.jpg output.jpg
# Gilad Cohen  203722442
# Ofek Talker  311369961

import sys
import cv2
import numpy as np
from datetime import datetime
size = 800
"""
PipeLine:
        readImage -> Blurring -> GrayScale -> ThresHolding -> Denoising -> 
            Canny-Edge -> Find Contours -> Find Corners co-ordinates -> Crop  the contour ->
                Sharping & Brightness correction
                
STEPS:
    1. Read from terminal
    2. Blurring - Gaussian Filter
    3. image to Gray scale
    4. ThresHolding - simple or adaptive Thresholding
    5. Denoising - cv2.fastNlMeansDenoising
    6. Canny Edge detection
    7. Find Contours
"""

def mapp(h):
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
    img = cv2.resize(img, (size, size))
    return img


# for terminal execute
path_img = sys.argv[1]
path_output = sys.argv[2]

# # check runtime from the beginning
# start_time = datetime.now()
# start_time = start_time.strftime("%H:%M:%S")
# print("Start Time: " + start_time)
# start_time = datetime.now()

# read image in grayScale
image = read_image(path_img)
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



blurred=cv2.GaussianBlur(gray, (5, 5), 1)
threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
cv2.imshow("Blur", blurred)

edged = cv2.Canny(threshold, 30, 70, apertureSize=7)  #30 MinThreshold and 50 is the MaxThreshold
cv2.imshow("Canny", edged)


contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours = sorted(contours, key=cv2.contourArea, reverse=True)

#Extracts contours of the page
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)

    if len(approx) == 4:
        target = approx
        break
approx = mapp(target) #find endpoints

pts = np.float32([[0, 0], [size, 0], [size, size], [0, size]])

op = cv2.getPerspectiveTransform(approx, pts)  #bird eye view effect
dst = cv2.warpPerspective(orig, op, (size, size))



#ret1,dst = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
cv2.imwrite(path_output, dst)
cv2.imshow("Scanned", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# total run time
end_time = datetime.now()
total_time = datetime.now()
total_time = end_time - start_time
end_time = end_time.strftime("%H:%M:%S")
total_time = str(total_time)
print("End time: " + end_time)
print("Total time: " + total_time)
