import requests

# Endpoint URL
url = "https://6e65-119-158-64-26.ngrok-free.app/analyze_video/"

# Path to the video file
video_path = "/home/scenescribe/Desktop/scenescribe/avis/video_20250415_103710.avi"

# Prompt to send
prompt_text = "Describe what's happening in this video, be very concise and precise"

# Open the video file in binary mode
with open(video_path, "rb") as video_file:
    files = {
        "video": video_file,
    }
    data = {"prompt": prompt_text}

    # Send the POST request
    response = requests.post(url, files=files, data=data)

# Print the response
print("Status Code:", response.status_code)
print("Response Text:", response.text)
