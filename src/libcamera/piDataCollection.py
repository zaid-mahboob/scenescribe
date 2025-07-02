import time
import os
from picamera2 import Picamera2
import firebase_admin
from firebase_admin import db

# Initialize Firebase
cred_obj = firebase_admin.credentials.Certificate(
    "/home/scenescribe/Desktop/scenescribe/credentials/credentials.json"
)
default_app = firebase_admin.initialize_app(
    cred_obj,
    {
        "databaseURL": "https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app"
    },
)

# Initialize the Picamera2 object
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
picam2.set_controls({"AfMode": 2})
picam2.set_controls({"AfTrigger": 0})

# Directory to save images
image_dir = "/home/scenescribe/Desktop/scenescribe/testimages"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


# Function to capture and save an image
def capture_image():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(image_dir, f"image_{timestamp}.jpg")
    picam2.capture_file(image_path)
    print(f"Captured image: {image_path}")
    return image_path


# Fetch the initial button state from Firebase
ref = db.reference("/intValue")
previous_button_state = ref.get()

# Main loop to poll Firebase for button state and capture image
while True:
    # Fetch the current button state from Firebase
    current_button_state = ref.get()

    # Check if the button state has changed (pressed)
    if current_button_state != previous_button_state:
        if current_button_state == 1:  # Assuming 1 means button pressed

            picam2.set_controls({"AfTrigger": 0})
            time.sleep(1)
            print("Button pressed!")

            # Capture and save image
            image_path = capture_image()

            # Optionally, upload to Firebase or store metadata if needed
            ref.set(f"Captured image: {image_path}")

        # Update the previous button state to the current one
        previous_button_state = current_button_state

    # Wait a small amount of time to reduce constant polling load
    time.sleep(1)  # Adjust polling frequency if necessary
