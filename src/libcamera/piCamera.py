from picamera2 import Picamera2
import numpy as np
import cv2
import time

# Initialize Picamera2
picam2 = Picamera2()

# Configure camera settings
camera_config = picam2.create_preview_configuration(main={"size": (3840, 2160)})
picam2.configure(camera_config)

# Start the camera
picam2.start()
time.sleep(2)  # Allow the camera to adjust exposure & focus

# Enable continuous autofocus
picam2.set_controls({"AfMode": 2})  # Continous focus mode
# picam2.set_controls({"AfTrigger": 0})  # Manual focus mode

num_frames = 7  # Number of frames to capture
best_sharpness = 0
best_image = None

def calculate_sharpness(image):
    """Calculate image sharpness using the Laplacian variance method."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

print("Capturing images...")
start_time = time.time()
for i in range(num_frames):
    # Capture frame as numpy array
    frame = picam2.capture_array("main")
    
    # Calculate sharpness
    sharpness = calculate_sharpness(frame)
    print(f"Frame {i+1} - Sharpness: {sharpness}")

    # Save the sharpest frame
    if sharpness > best_sharpness:
        best_sharpness = sharpness
        best_image = frame
    print(f"best sharpness: {best_sharpness}")

    time.sleep(0.5)  # Allow autofocus to adjust

# Save the best image
if best_image is not None:
    best_image = cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
    cv2.imwrite("best_autofocus.jpg", best_image)
    print("Best autofocus image saved as best_autofocus.jpg")
    print(f"image taking time: {time.time() - start_time }")

best_sharpness = 0
best_image = None
time.sleep(5)
for i in range(num_frames):
    # Capture frame as numpy array
    frame = picam2.capture_array("main")
    
    # Calculate sharpness
    sharpness = calculate_sharpness(frame)
    print(f"Frame {i+1} - Sharpness: {sharpness}")

    # Save the sharpest frame
    if sharpness > best_sharpness:
        best_sharpness = sharpness
        best_image = frame

    time.sleep(0.5)  # Allow autofocus to adjust

# Save the best image
if best_image is not None:
    best_image = cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
    cv2.imwrite("best_autofocus2.jpg", best_image)
    print("Best autofocus image saved as best_autofocus.jpg")
picam2.stop()
'''
from picamera2 import Picamera2, Preview
import time

for i in range(1):
    # Initialize the camera
    picam2 = Picamera2()

    # Configure camera for still image capture
    config = picam2.create_still_configuration(main={"size": (8000, 6000 )})
    picam2.configure(config)
    
    # picam2.set_controls({"LensPosition": (6.9 + 0.1*i)})  # Adjust focus (try different values)
    picam2.start()
    time.sleep(2)
    picam2.set_controls({"AfMode": 1})  # Manual focus mode
    picam2.set_controls({"AfTrigger": 0})  # Manual focus mode
    time.sleep(5)  # Allow camera to initialize

    # Capture the image
    # picam2.capture_file(f"image_8k_manualfocus_{6.9 + 0.1*i}.png")
    picam2.capture_file(f"image_conitnousfocus.jpg")
    print("Image captured with manual focus")
    # time.sleep(5)
    # picam2.capture_file("image_autofocus2.jpg")

    picam2.close()

'''
