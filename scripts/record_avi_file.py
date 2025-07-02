import time
import os
import cv2
from picamera2 import Picamera2
import numpy as np

# === Setup directories ===
video_dir = "/home/scenescribe/Desktop/scenescribe/avis"  # Change if needed
os.makedirs(video_dir, exist_ok=True)

# === Initialize Picamera2 ===
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
# picam2.set_controls({"AfMode": 2})
# picam2.set_controls({"AfTrigger": 0})
picam2.set_controls({"AfMode": 2})
# === Video recording parameters ===
fps = 3  # Lower FPS = smaller file size
duration = 10  # seconds
frame_size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Compressed codec

# === Output file ===
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"video_smolvlm.avi"
filepath = os.path.join(video_dir, filename)
out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

print(f"Recording video to: {filepath}")

# === Capture frames ===
start_time = time.time()
while time.time() - start_time < duration:
    picam2.set_controls({"AfMode": 2})
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
    out.write(frame)
    time.sleep(1 / fps)

# === Cleanup ===
out.release()
picam2.stop()
print("Recording finished.")
