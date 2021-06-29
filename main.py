import cv2 as cv
import numpy as np
from math import atan2,pi,degrees

from scipy.ndimage.interpolation import rotate
img = cv.imread("data/Capturar.png")
light = cv.imread("data/light.png", -1)


shape_x,shape_y,_ = light.shape

print(shape_x,shape_y)

blurred = cv.GaussianBlur(img, (5, 5), 0)
mask = np.zeros(img.shape, np.uint8)
kernel = np.ones((9,9),np.uint8)
hsv = cv.cvtColor(blurred,cv.COLOR_BGR2HSV)##
lower_range = np.array([25, 50, 150])
upper_range = np.array([110,150,255])
mask_img = cv.inRange(hsv,lower_range,upper_range)

#mask_img =cv.bitwise_not(mask_img)
contours, hierarchy = cv.findContours(mask_img,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(mask, contours, -5, (0,255,250),10)
cv.drawContours(mask, contours, -5, (0,255,250),10)
output = np.where(mask==np.array([0, 255, 250]), blurred, img)
cv.drawContours(output, contours, -5, (0,255,0),10)
blurred = cv.GaussianBlur(output, (25, 25), 0)
lower_range = np.array([25, 60, 150])
upper_range = np.array([110,150,255])
mask_img = cv.inRange(hsv,lower_range,upper_range)
cv.drawContours(mask, contours, -5, (0,255,250),20)
cv.drawContours(output, contours, -5, (250,255,50),5)

cv.imshow("img",output )
cv.waitKey(0)

output = np.where(mask==np.array([0, 255, 250]), blurred, output)

cv.imshow("img",output )
cv.waitKey(0)
