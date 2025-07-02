import base64
import requests
from picamera2 import Picamera2, Preview
import time
import os
from openai import OpenAI
import pyttsx3

# OpenAI API Key
from dotenv import load_dotenv

if not os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key is missing. Please set it in the environment variable or directly in the script.")
else:
    print("Welcome! Type 'listen brother' to start a conversation.")
    
client = OpenAI(os.getenv("OPENAI_API_KEY"))

picam2 = Picamera2()
picam2.start()

# Allow the camera to warm up
time.sleep(2)

# Capture an image and save it to a buffer
image_path = "/home/scenescribe/Desktop/scenescribe/test.jpg"
picam2.capture_file(image_path)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Getting the base64 string
base64_image = encode_image(image_path)

# Initialize the text-to-speech converter
converter = pyttsx3.init()

# Prepare the request headers and payload
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {client}"  # Ensure api_key is defined
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

# Send the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Process and speak the response
if response.status_code == 200:
    text = response.json()['choices'][0]['message']['content']
    print(text)
    converter.say(text)
    converter.runAndWait()
else:
    print(f"Error: {response.status_code}, {response.text}")