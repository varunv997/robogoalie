import numpy as np
import cv2

left = cv2.imread('test1.jpg')
resized1 = cv2.resize(left,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
right = cv2.imread('test2.jpg')
resized2 = cv2.resize(right,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

#hsvs

hsv1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2HSV)

#mask colors
lower_blue = np.array([99, 159, 162])
upper_blue = np.array([107, 187, 255])

mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
mask2 = cv2.inRange(hsv2, lower_blue, upper_blue)

#erosion
kernel = np.ones((5,5), np.uint8)
eroded1 = cv2.erode(mask1, kernel, iterations = 1)
eroded2 = cv2.erode(mask2, kernel, iterations = 1)

#dilation
dilated1 = cv2.dilate(eroded1, kernel, iterations = 3)
dilated2 = cv2.dilate(eroded2, kernel, iterations = 3)

#finding contour
ret1,thresh1 = cv2.threshold(dilated1,127,255,0)
_, contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
ret2, thresh2 = cv2.threshold(dilated2, 127,255,0)
_, contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(resized1, contours1, -1, (0,255,0), 3)
cv2.drawContours(resized2, contours2, -1, (0,255,0), 3)

#find the largest contour just in case:
lc1 = contours1[0]
lc2 = contours2[0]
for i in contours1:
	if (cv2.contourArea(lc1) > cv2.contourArea(i)):
		lc1 = i

for j in contours2:
	if (cv2.contourArea(lc2) > cv2.contourArea(j)):
		lc2 = j

rect1 = cv2.minAreaRect(lc1)
rect2 = cv2.minAreaRect(lc2)

#take the mid pt
center1x = rect1[0][0]
center1y = rect1[0][1]
center2x = rect2[0][0]
center2y = rect2[0][1]


#cv2.imshow('testimg 1', resized1)
#cv2.imshow('testimg 2', resized2)
cv2.waitKey(0)
cv2.destroyAllWindows()
