from picamera2 import Picamera2
import time
import numpy as np
import cv2

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration())

# Start the camera preview (in background)
picam2.start()

# Open a window to display the feed
while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Convert the frame to BGR format for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()
