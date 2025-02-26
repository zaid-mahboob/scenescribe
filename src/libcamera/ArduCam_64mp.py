import libcamera
import time

def capture_image():
    # Initialize camera manager
    camera_manager = libcamera.CameraManager.singleton()
    
    # Get available cameras
    if len(camera_manager.cameras) == 0:
        print("No cameras found!")
        return

    # Select the first available camera
    camera = camera_manager.get(camera_manager.cameras[0].id)

    # Acquire camera control
    camera.acquire()

    # Configure camera (set resolution to match `libcamera-still`)
    config = camera.generate_configuration([libcamera.StreamRole.StillCapture])
    stream_config = config.at(0)  # Use .at(0) instead of [0]

    # Set resolution and format
    stream_config.pixel_format = libcamera.PixelFormat("RGB888")  # Correct way to set pixel format
    stream_config.size.width = 4624
    stream_config.size.height = 3472
    
    # Apply the configuration
    camera.configure(config)

    # Enable autofocus
    controls = {
        libcamera.controls.AfMode: libcamera.controls.AfModeEnum.Auto,
        libcamera.controls.AfTrigger: libcamera.controls.AfTriggerEnum.Start
    }

    # Start camera
    camera.start(controls)

    # camera.controls.set(controls)
    # Allow time for autofocus to adjust
    time.sleep(2)  # Adjust timing as needed

    # Capture an image WIP
    request = camera.create_request()
    buffer = request.findBuffer()
    image_data = buffer.map()

    # Save the image as a file
    with open("captured_image.jpg", "wb") as f:
        f.write(image_data)

    print("Image saved as captured_image.jpg")

    # Stop camera and release resources
    camera.stop()
    camera.release()

# Run the capture function
capture_image()
