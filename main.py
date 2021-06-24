import cv2 as cv
import numpy as np

img = cv.imread("data/saber.jpg")
blurred = cv.GaussianBlur(img, (15, 15), 0)

kernel = np.ones((9,9),np.uint8)

hsv = cv.cvtColor(blurred,cv.COLOR_BGR2HSV)##
lower_range = np.array([25, 50, 150])
upper_range = np.array([100,150,255])
mask_img = cv.inRange(hsv,lower_range,upper_range)
contours, hierarchy = cv.findContours(mask_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0,255,0), 3)

cv.imshow("img",img)

cv.waitKey(0)

