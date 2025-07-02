from openai import OpenAI
import sys
import cv2
import os
import base64
from picamera2 import Picamera2, Preview
import speech_recognition as sr
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import threading
import time
import copy
import joblib
from utils.misc import extract_positions_from_string, generate_navigation
import whisper
import time
import sounddevice as sd
import numpy as np
import wave
import webrtcvad
import firebase_admin
from firebase_admin import db
from piper.voice import PiperVoice
from dotenv import load_dotenv
import noisereduce as nr
import scipy.io.wavfile as wavfile
import requests
from ollama import chat
from ollama import ChatResponse
import re


language = "Urdu"

import pygame
import time
pygame.mixer.init()

if language == None:
    print("Using English language as default")
else:
    print(f"Using {language} Language")
    time.sleep(2)

cred_obj = firebase_admin.credentials.Certificate('credentials/credentials.json')
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL':'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

# Load the saved model and vectorizer
loaded_model = joblib.load("models/nb_classifier_3_classes_v2.pkl")
loaded_vectorizer = joblib.load("models/vectorizer_3_classes_v2.pkl")

# Initialize the recognizer
sys.stderr = open(os.devnull, 'w')
recognizer = sr.Recognizer()

print("Initializing Camera")

picam2 = Picamera2()
picam2.start()
time.sleep(0.1)

print("Camera Initialized")

text = ""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image():
    image_path = "/home/scenescribe/Desktop/scenescribe/test.jpg"
    picam2.capture_file(image_path)
    return image_path

conversation_history = [{
    "role": "system",
    "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful."}]

video_prompt = "Please explain what is happening in the video!"
endpoint_url = "https://674b-59-103-72-42.ngrok-free.app/analyze_video/"
openai_key = ""
if not openai_key:
    print("OpenAI API key is missing. Please set it in the environment variable or directly in the script.")
else:
    print("Welcome! Type 'listen brother' to start a conversation.")
# print(os.getenv("OPENAI_API_KEY"))
openai = OpenAI(api_key=openai_key)
# OpenAI(api_key=)
print("Reached this bullet point")

user_text = None
lock = threading.Lock()  # To ensure thread-safe updates to `user_text`

model = whisper.load_model("tiny")
print(f"Text to speech model loaded!")

# Recording settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
AUDIO_FILE = "recorded_audio.wav"  # Output file
FRAME_DURATION = 30  # Frame size in milliseconds
VAD = webrtcvad.Vad(3)  # Aggressiveness level (0-3): 3 is most sensitive

recognizer = sr.Recognizer()  # Initialize SpeechRecognizer for noise adjustment

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.button_on = False

    def set_button_state(self, state: bool):
        with self.lock:
            self.button_on = state

    def get_button_state(self) -> bool:
        with self.lock:
            return self.button_on

shared_state = SharedState()

# --- Firebase Polling Thread ---
def firebase_polling_loop(shared: SharedState):
    while True:
        try:
            value = db.reference("/intValue").get()
            shared.set_button_state(value == 1)
        except Exception as e:
            print(f"Firebase read error: {e}")
        time.sleep(0.1)

def denoise_wav(file_path):
    print(f"🔧 Denoising audio: {file_path}")
    rate, data = wavfile.read(file_path)
    if len(data.shape) == 2:
        data = np.mean(data, axis=1).astype(np.int16)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    if reduced_noise.dtype != np.int16:
        reduced_noise = np.clip(reduced_noise, -32768, 32767).astype(np.int16)
    wavfile.write(file_path, rate, reduced_noise)
    print("✅ Noise reduction complete.")

# --- Unified Recording Function ---
def record_with_firebase_control(output_path="recorded_audio.wav", sample_rate=16000, frame_duration=30):
    print("🔁 Waiting for Firebase button to turn ON...")

    # Start Firebase thread if not already running
    if not any([t.name == "FirebaseThread" for t in threading.enumerate()]):
        firebase_thread = threading.Thread(target=firebase_polling_loop, args=(shared_state,), daemon=True, name="FirebaseThread")
        firebase_thread.start()

    # Wait for button ON
    while not shared_state.get_button_state():
        time.sleep(0.1)

    print("🎙️ Recording started...")
    buffer = []

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
        while shared_state.get_button_state():
            audio_frame, _ = stream.read(int(sample_rate * frame_duration / 1000))
            buffer.append(audio_frame)

    print("⏹️ Recording stopped. Saving...")

    audio_data = np.concatenate(buffer, axis=0)
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    denoise_wav(output_path)

    print(f"✅ Finished. Audio saved and denoised at: {output_path}")


def convert_and_play_speech(text):
    try:
        # Load the voice model
        model = "models/tts/en_GB-northern_english_male-medium.onnx"
        voice = PiperVoice.load(model)

        output_file = "output.wav"
        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050)  # Sample rate
            voice.synthesize(text, wav_file)

        # Read and play
        wav_data = np.fromfile(output_file, dtype=np.int16)
        sd.play(wav_data, 22050)
        sd.wait()
        print("Playback finished.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def openai_convert_and_play_speech(text):
    try:
        # Load the voice model
        start_time = time.time()
        speech_file_path = "output.wav"
        response = openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=text
        )
        response.stream_to_file(speech_file_path)

        # Read and play
        sound = pygame.mixer.Sound(speech_file_path)

        # Play the sound
        sound.play()

        # Keep the script running until the audio finishes playing
        time.sleep(sound.get_length())
        print("Playback finished.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
# Wait until the audio finishes playing


ground_floor_hierarchical_tree = {
    "Department Entrance": {
        "In Front": {
            "Reception": "Central point connecting all branches",
            "Stairs": "Leads to the upper floor"
        },
        "Right Turn": {
            "Classroom 1": "1st room on the right side",
            "Industrial Automation Lab": "1st room on the left side",
            "Robotics Lab": "2nd room on the left side",
            "HOD Corridor": "on the Straight at the end of the corridor",
            "Secondary Exit": "on the Right side on end of the corridor"
        },
        "Left Turn": {
            "CAD/CAM Lab": "1st room on the right side",
            "Machine Vision Lab": "1st room on the left side",
            "Electronics Lab": "2nd room on the right side",
            "Washroom": "on the Left side on the end of the corridor",
            "Second stairs": "on the Left side on the end of the corridor"
        },
    }
}

def classify_input(sentence):
    sentence_transformed = loaded_vectorizer.transform([sentence])
    prediction = loaded_model.predict(sentence_transformed)
    print(f"The query '{sentence}' is classified as: {prediction[0]}")
    return prediction[0]


def record_video_from_camera(duration=5, fps=2, output_filename="video_smolvlm.avi",
                              resolution=(640, 480), video_dir="/home/scenescribe/Desktop/scenescribe/avis"):
    """
    Records video from a pre-initialized Picamera2 object.

    Args:
        picam2: Initialized Picamera2 object.
        duration (int): Duration in seconds to record.
        fps (int): Frames per second.
        output_filename (str): Name of the AVI file to save.
        resolution (tuple): Frame size (width, height).
        video_dir (str): Directory to save video.

    Returns:
        str: Full path to the saved video file.
    """

    # Ensure save directory exists
    os.makedirs(video_dir, exist_ok=True)

    # Prepare output path and writer
    filepath = os.path.join(video_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, resolution)

    print(f"📹 Recording video to: {filepath}")

    start_time = time.time()
    while time.time() - start_time < duration:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        time.sleep(1 / fps)

    # Cleanup
    out.release()
    print("✅ Recording finished.")

    return filepath

def analyze_video_with_prompt(video_path, prompt_text, endpoint_url):
    """
    Sends a video and prompt to an analysis API endpoint and returns the response.

    Args:
        video_path (str): Full path to the video file.
        prompt_text (str): Instruction or question for the model.
        endpoint_url (str): URL of the analysis API.

    Returns:
        dict: Response from the server containing status and text.
    """
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {'prompt': prompt_text}

            response = requests.post(endpoint_url, files=files, data=data)

            return {
                "status_code": response.status_code,
                "response_text": response.text
            }

    except Exception as e:
        return {
            "status_code": None,
            "response_text": f"Error: {e}"
        }

def activity_detection(user_input):
    try:
        # Attempt to record the video from the camera
        video_path = record_video_from_camera()

        # Attempt to analyze the video and get the response
        response = analyze_video_with_prompt(video_path=video_path, prompt_text=user_input, endpoint_url=endpoint_url)
        # print(response)

        # Ensure the response contains the expected key
        if "response_text" not in response:
            raise ValueError("Response does not contain 'response_text' key.")

        full_response = response["response_text"]
        

        print(f"Full Response: {full_response}")

        # Use regex to extract the assistant's answer from the response text
        match = re.search(r'Assistant: (.*?)"\s*,\s*"processing_time_seconds', full_response)

        if match:
            # Decode the assistant's response correctly
            assistant_answer = match.group(1).encode('utf-8').decode('unicode_escape')
            print(f"Assistant's Answer: {assistant_answer}")
            agent_1_output = full_response
            assistant_answer = video_agent_2(user_input, assistant_answer)
            print(f"Final answer is : {assistant_answer}")
            return assistant_answer
        else:
            raise ValueError("Could not find the assistant's answer in the response text.")

        return False

    except Exception as e:
        # Handle any exceptions that occur during the process
        print(f"An error occurred: {e}")
        return False

def video_agent_2(user_input, agent_1_output):
    # Customize the prompt for Agent 2
    prompt = f"""
    I am visually disabled. You
    are an assistant for individuals with visual disability. Your
    role is to shrink the given information into a couple
    of lines in order to reduce the cognitive overloading. Your
    task is to remove all the unnecessary information from the
    given given information. Only keep information that is relevant
    to this query {user_input} Don’t mention that I am visually
    disabled to offend me, or that many details that he feels that
    he wishes he could see Avoid extra information like type kinds
    category so he felt disabled for not able to judge itself. since
    he’s blind so don’t start like this image or in the image and
    remove extra information that is not required to tell the blind.
    don’t add information by which I had to use my eyes and I
    feel disabled. Scene Description: {agent_1_output}.
    """
    if language is not None:
        prompt = prompt + " " + f"Please give final ouput in {language} language"
    messages = {"role": "user", "content": prompt}
    messages_ = {"role": "user", "content": user_input}
    print("Content Prepared")

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # Call OpenAI API with image and text input, including conversation history
    conversation_history.append(messages_)
    completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=temp_history
            )
    output = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": output})
    # print(conversation_history)

    # converter.say(output)
    # converter.runAndWait()
    return output

def deepseek(user_input):
    try:
        response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue? Be concise and precise',
        },
        ])
        text = response['message']['content']
        # or access fields directly from the response object
        # print(response.message.content)

        match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
        if match:
            main_part = match.group(1).strip()
            return main_part
        else:
            return None
    except Exception as e:
        print(f"Error Occured: {e}")

def explanation_agent_1(image_base64, user_input):
    # Customize the prompt for Agent 1
    prompt = f"""
    “I am visually disabled. You are an
    assistant for individuals with visual disability. Your role is
    to provide helpful information and assistance based on my
    query. Your task is to {user_input}. Don’t mention that I
    am visually disabled or extra information to offend me. Be
    straightforward with me in communicating and don’t add any
    future required output, tell me what asked only
    """
    messages = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
    
    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # print(temp_history)
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=temp_history
            )
            
    # Get the AI's response content
    response_content = completion.choices[0].message.content

    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content

def explanation_agent_2(user_input, agent_1_output):
    # Customize the prompt for Agent 2
    prompt = f"""
    I am visually disabled. You
    are an assistant for individuals with visual disability. Your
    role is to shrink the given information into a couple
    of lines in order to reduce the cognitive overloading. Your
    task is to remove all the unnecessary information from the
    given given information. Only keep information that is relevant
    to this query {user_input} Don’t mention that I am visually
    disabled to offend me, or that many details that he feels that
    he wishes he could see Avoid extra information like type kinds
    category so he felt disabled for not able to judge itself. since
    he’s blind so don’t start like this image or in the image and
    remove extra information that is not required to tell the blind.
    don’t add information by which I had to use my eyes and I
    feel disabled. Always return solution in a form of a paragraph in a way that 
    someone is talking to another person. Scene Description: {agent_1_output}.
    """
    if language is not None:
        prompt = prompt + " " + f"Please give final ouput in {language} language"
    messages = {"role": "user", "content": prompt}
    messages_ = {"role": "user", "content": user_input}
    print("Content Prepared")

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # Call OpenAI API with image and text input, including conversation history
    conversation_history.append(messages_)
    completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=temp_history
            )
    output = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": output})
    # print(conversation_history)

    # converter.say(output)
    # converter.runAndWait()
    return output

def navigation_agent_1(image_base64, user_input):
    # Customize the prompt for Agent 1
    prompt = f"""Provide valid json output. I am visually disabled. You are an
    navigation assistant for individuals with visual disability. Your role is
    to provide navigation and direction assistance for my input
    query. My query is {user_input}, help me using the image, don’t add any
    future required output, tell me what asked only. You have to guide me the user in terms of navigation telling in which direction should they move
    how many estimated steps are needed to reach the destination, if the destination is not so clear in the image, use your common sense,
    to judge how a human will use his/her brain with the given image to decide what should be the logical navigation and direction for reaching end goal.
    Reminder that you need to navigate the person as per his requirements not to chit chat and don't use that you can't help, you are the only source
    Give directions in term of weather should I go forward, left, right, etc. Can you please also tell an angle at which I need to walk, to reach my destination.
    In case where you are not sure about something, use common sense to guide.
    """
    messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
                model="gpt-4o-2024-05-13",
                response_format={ "type": "json_object" },
                messages=messages
            )

            # Get the AI's response content
    response_content = completion.choices[0].message.content
    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content

def navigation_agent_2(user_input, agent_1_output):
    # Customize the prompt for Agent 2
    prompt = f"""
    I am visually disabled. You
    are an assistant for individuals with visual disability. Your
    role is to shrink the given information into a couple
    of lines in order to reduce the cognitive overloading. Your
    task is to remove all the unnecessary information from the
    given given information. Only keep information that is relevant
    to this query {user_input} Don’t mention that I am visually
    disabled to offend me, or that many details that he feels that
    he wishes he could see Avoid extra information like type kinds
    category so he felt disabled for not able to judge itself. since
    he’s blind so don’t start like this image or in the image and
    remove extra information that is not required to tell the blind.
    don’t add information by which I had to use my eyes and I
    feel disabled. Scene Description: {agent_1_output}.
    """
    if language is not None:
        prompt = prompt + " " + f"Please give final ouput in {language} langauge"
    messages = [{"role": "user", "content": prompt}]
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
    output = completion.choices[0].message.content
    # converter.say(output)
    # converter.runAndWait()
    return output

def global_navigation_agent(user_input, tree):
    # Customize the prompt for Agent 1
    prompt = f"""
    “You are an assistant of visually impaired people, your task is to take user input and return only two things, one would be initial position and other
    final posiion. You are given user query and a hierarchical tree which represents a map of a building, you need to find the most optimum and logical position
    based on the user query and tree.

    You only have to give answer in a json format:

    {{
        "initial_position" = "",
        "final_position" = ""
    }}

    Tree is given below:
    {tree}

    User Query is this: {user_input}
    """
    messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }]
    # print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Get the AI's response content
    response_content = completion.choices[0].message.content
    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content

def wait_for_listen_command():
    
    try:
            print("Initializing microphone...")
            # recognizer.adjust_for_ambient_noise(source, 2)  # Adjust for ambient noise
            try:
                record_with_firebase_control(AUDIO_FILE, SAMPLE_RATE)
                # Transcribe the recorded audio
                start_time = time.time()
                # result = model.transcribe(AUDIO_FILE)
                audio_file= open(AUDIO_FILE, "rb")
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language= "en"
                )
                end_time = time.time()
                result= transcription.text

                # Print the transcribed text and processing time
                print(f"? Processing Time: {end_time - start_time:.2f} seconds")
                print("?? Transcribed Text:", result)
                return result

            except sr.UnknownValueError:
                # Handle case where speech was not understood
                print("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                # Handle API request issues
                print(f"Could not request results from Google Speech Recognition service; {e}")
                  # Exit the loop if there's a serious issue
            except sr.WaitTimeoutError:
                # Handle timeout waiting for speech
                print("Listening timed out. Retrying...")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main loop for interaction
def display_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    plt.imshow(image)
    plt.axis("off")
    plt.show()



# Main loop
while True:
    print("Starting...")
    # user_input = "Please explain my surrounding in detail"
    # user_input = input("Please enter the query: ")
    user_input = wait_for_listen_command()
    print(user_input)
    print("Command Captured")
    if (True):
        # classification_result = "Scene Explanation"
        classification_result = classify_input(user_input)
        if ("detail" in user_input.lower()):
            print("Using Video Analysis")
            output = activity_detection(user_input)
            # if output != False:
            openai_convert_and_play_speech(output)
        elif (classification_result == "Scene Explanation"):
            print("Capturing image...")
            # img_path = "/home/scenescribe/Desktop/scenescribe/test6.jpg"
            img_path = get_image()
            base64_image = encode_image(img_path)
            print("Processing with Agent 1...")
            agent_1_output = explanation_agent_1(base64_image, user_input)
            print(f"Agent 1 Output: {agent_1_output}")
            print("Processing with Agent 2...")
            agent_2_output = explanation_agent_2(user_input,agent_1_output)
            print(f"Agent 2 Output: {agent_2_output}")
            openai_convert_and_play_speech(agent_2_output)
        else:
            print("Capturing image...")
            img_path = get_image()
            base64_image = encode_image(img_path)
            print("Processing with Agent 1...")
            agent_1_output = navigation_agent_1(base64_image, user_input)
            print(f"Agent 1 Output: {agent_1_output}")
            print("Processing with Agent 2...")
            agent_2_output = navigation_agent_2(user_input,agent_1_output)
            print(f"Agent 2 Output: {agent_2_output}")
            openai_convert_and_play_speech(agent_2_output)
    elif user_input.lower() == "exit":
        print("Exiting the assistant.")
        break
    else:
        print("Say 'listen' to activate the assistant or say 'exit' to quit.")