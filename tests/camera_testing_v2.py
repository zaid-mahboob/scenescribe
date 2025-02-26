from picamera2 import Picamera2
import time
import cv2

#Capture image using picamera module
camera = Picamera2()
camera.start()
time.sleep(0.1)
camera.capture_file('test3.jpg')

#Read image from MicroSD card
image = cv2.imread('test3.jpg', -1)

#Display image in an OpenCV window
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 640, 480)
cv2.imshow('Image', image)
cv2.waitKey(0)