# run - Scanner.py 1/page.jpg output.jpg
# Gilad Cohen  203722442
# Ofek Talker  311369961

import sys
import cv2
import numpy as np
from datetime import datetime

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def read_image(path):
    img = cv2.imread(path)
    img=cv2.resize(img,(800,800))
    return img


# for terminal execute
path_img = sys.argv[1]
path_output = sys.argv[2]

# check runtime from the beginning
start_time = datetime.now()
start_time = start_time.strftime("%H:%M:%S")
print("Start Time: " + start_time)
start_time = datetime.now()

# read image in grayScale
image = read_image(path_img)
#image = cv2.resize(image, (1500, 880))
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
cv2.imshow("Blur",blurred)

edged=cv2.Canny(blurred,0,50)  #30 MinThreshold and 50 is the MaxThreshold
cv2.imshow("Canny",edged)


contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours=sorted(contours,key=cv2.contourArea,reverse=True)

#the loop extracts the boundary contours of the page
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapp(target) #find endpoints of the sheet

pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  #map to 800*800 target window

op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
dst=cv2.warpPerspective(orig,op,(800,800))


cv2.imshow("Scanned",dst)
# press q or Esc to close
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path_output, dst)

# total run time
end_time = datetime.now()
total_time = datetime.now()
total_time = end_time - start_time
end_time = end_time.strftime("%H:%M:%S")
total_time = str(total_time)
print("End time: " + end_time)
print("Total time: " + total_time)
