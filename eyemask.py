import cv2
import numpy as np

def nothing(x):
	pass


cv2.namedWindow('Color Capture')

#Create TrackBars
cv2.createTrackbar('M1', 'Color Capture', 0, 255, nothing)
cv2.createTrackbar('S1', 'Color Capture', 0, 255, nothing)
cv2.createTrackbar('V1', 'Color Capture', 0, 255, nothing)
cv2.createTrackbar('M2', 'Color Capture', 0, 255, nothing)
cv2.createTrackbar('S2', 'Color Capture', 0, 255, nothing)
cv2.createTrackbar('V2', 'Color Capture', 0, 255, nothing)

frame = cv2.imread('test2.jpg')
frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

while(1):
	
	cv2.imshow('Color Capture', frame)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	
	m1 = cv2.getTrackbarPos('M1', 'Color Capture')
	s1 = cv2.getTrackbarPos('S1', 'Color Capture')
	v1 = cv2.getTrackbarPos('V1', 'Color Capture')

	m2 = cv2.getTrackbarPos('M2', 'Color Capture')
	s2 = cv2.getTrackbarPos('S2', 'Color Capture')
	v2 = cv2.getTrackbarPos('V2', 'Color Capture')

	lower_color=np.array([m1, s1, v1])
	upper_color=np.array([m2, s2, v2])
	#apply mask
	mask = cv2.inRange(hsv, lower_color, upper_color)
	res = cv2.bitwise_and(frame, frame, mask=mask)
	#show the result
	cv2.imshow('res', res)
file.close()
cap.release()
cv2.destroyAllWindows()
