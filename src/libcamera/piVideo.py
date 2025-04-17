from picamera2 import Picamera2
import cv2
import time

# Initialize Picamera2

picam2 = Picamera2()

# Configure camera for preview mode (video)
# video_config = picam2.create_video_configuration(main={"size": (1920, 1080)})  # Adjust resolution as needed
# picam2.configure(video_config)

# Start the camera
picam2.start()
time.sleep(2)  # Allow camera to initialize

# picam2.set_controls({"AfTrigger": 0})

print("Press 'q' to exit the video window.")
i=0
while True:
    # picam2.set_controls({"AfMode": 2})  # Continous focus mode
    # picam2.set_controls({"AfTrigger": 0})
    # Capture frame as numpy array
    frame = picam2.capture_array()

    # Convert frame to BGR for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    i=i+1

    # Show video feed
    cv2.imshow("Arducam HawkEye Live Video", frame)
    # timestamp = int(time.time())
    # cv2.imwrite(f"./Desktop/dataset_v3/imageG_1080_{timestamp}.png",frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()

