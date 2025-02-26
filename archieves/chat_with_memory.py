from openai import OpenAI
import sys
import cv2
import os
import base64
from picamera2 import Picamera2, Preview
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv

# Initialize the recognizer
sys.stderr = open(os.devnull, 'w')
converter = pyttsx3.init()
recognizer = sr.Recognizer()

picam2 = Picamera2()
picam2.start()

print("Camera Initialized")

text = ""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to capture user text input
def get_text():
    text = ""  # Initialize text variable

    with sr.Microphone(device_index=1) as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio)
            output = f"You said {text}"
            print(output)
            converter.say(output)
            converter.runAndWait()
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    
    return text  # Always return text (could be an empty string if no speech is recognized)

# Function to capture an image using the webcam
def get_image():
    image_path = "/home/scenescribe/Desktop/scenescribe/test.jpg"
    picam2.capture_file(image_path)
    return image_path

if not os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key is missing. Please set it in the environment variable or directly in the script.")
else:
    print("Welcome! Type 'listen brother' to start a conversation.")
    
openai = OpenAI(os.getenv("OPENAI_API_KEY"))
# Initialize conversation memory
conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
print(conversation_history)

# Main loop for interaction
while True:
    # Get user input
    user_input = get_text()
    print(conversation_history)
    # Check for the trigger phrase
    if "listen" in user_input.lower():  # Kept this as True for easy testing
        converter.say("Lets start")
        converter.runAndWait()
        converter.say("Do you want to capture an image yes or no ")
        converter.runAndWait()
        text = get_text()
        
        if "yes" in text.lower():
            print("Here we go")
            img_path = get_image()  # Get image from predefined path

            # Encode the image to base64
            base64_image = encode_image(img_path)
            print("Image encoded")

            # Add image prompt to the conversation history
            api_pass_conversation = conversation_history.copy()
            conversation_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input}
                    # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })

            api_pass_conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })
            print("Content Prepared")
            # Call OpenAI API with image and text input, including conversation history
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_pass_conversation
            )
            
            # Get the AI's response content
            response_content = completion.choices[0].message.content
            converter.say(response_content)
            converter.runAndWait()

            # Add AI response to the conversation history
            conversation_history.append({"role": "assistant", "content": response_content})

        else:
            conversation_history.append({"role": "user", "content": user_input})
            # Call OpenAI API without image, using conversation history
            completion = openai.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history
            )

            # Get the AI's response content
            response_content = completion.choices[0].message.content
            print(f"AI: {response_content}")

            # Add AI response to the conversation history
            conversation_history.append({"role": "assistant", "content": response_content})

    elif user_input.lower() == "exit":
        print("Exiting the assistant.")
        break
    else:
        print("Say 'listen brother' to activate the assistant or type 'exit' to quit.")
