from openai import OpenAI
import sys
import cv2
import os
import base64
from picamera2 import Picamera2, Preview
import pyttsx3
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

cred_obj = firebase_admin.credentials.Certificate(
    "credentials/credentials.json")
default_app = firebase_admin.initialize_app(
    cred_obj,
    {
        "databaseURL": "https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app"
    },
)

# Load the saved model and vectorizer
loaded_model = joblib.load("models/nb_classifier_3_classes_v2.pkl")
loaded_vectorizer = joblib.load("models/vectorizer_3_classes_v2.pkl")

# Initialize the recognizer
sys.stderr = open(os.devnull, "w")
converter = pyttsx3.init()
recognizer = sr.Recognizer()

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
picam2.set_controls({"AfMode": 2})
picam2.set_controls({"AfTrigger": 0})

print("Camera Initialized")

text = ""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image():
    image_path = "/home/scenescribe/Desktop/scenescribe/test.jpg"
    picam2.capture_file(image_path)
    return image_path


conversation_history = [
    {
        "role": "system",
        "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful.",
    }
]

if not os.getenv("OPENAI_API_KEY"):
    print(
        "OpenAI API key is missing. Please set it in the environment variable or directly in the script."
    )
else:
    print("Welcome! Type 'listen brother' to start a conversation.")
# print(os.getenv("OPENAI_API_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# OpenAI(api_key=)
print("Reached this bullet point")

user_text = None
lock = threading.Lock()  # To ensure thread-safe updates to `user_text`

model = whisper.load_model("tiny")

# Recording settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
AUDIO_FILE = "recorded_audio.wav"  # Output file
FRAME_DURATION = 30  # Frame size in milliseconds
VAD = webrtcvad.Vad(3)  # Aggressiveness level (0-3): 3 is most sensitive

recognizer = sr.Recognizer()  # Initialize SpeechRecognizer for noise adjustment


def is_speech(audio_frame):
    """Check if the frame contains speech."""
    return VAD.is_speech(audio_frame.tobytes(), SAMPLE_RATE)


def record_audio(file_path, sample_rate):
    """Records audio from the microphone until silence is detected."""
    print("?? Listening... Speak now!")

    buffer = []
    silence_counter = 0
    # Stop recording after 10 silent frames (~0.3 sec)
    max_silence_frames = 100

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
        while True:
            audio_frame, _ = stream.read(
                int(sample_rate * FRAME_DURATION / 1000)
            )  # Read a small chunk
            buffer.append(audio_frame)

            if is_speech(audio_frame):
                silence_counter = 0  # Reset silence counter if speech is detected
                print("Speech Detected!  ")
            else:
                silence_counter += 1  # Increment silence counter

            if silence_counter > max_silence_frames:  # Stop if silence persists
                break

    print("? Speech detected. Processing...")

    # Convert buffer to NumPy array
    audio_data = np.concatenate(buffer, axis=0)

    # Save to WAV file
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def convert_and_play_speech(text):
    # Load the voice model
    model = "models/tts/en_GB-northern_english_male-medium.onnx"
    voice = PiperVoice.load(model)

    # Generate speech and save to .wav file
    with wave.open("output.wav", "w") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        # 22.05 kHz sample rate (adjust if needed)
        wav_file.setframerate(22050)
        audio = voice.synthesize(text, wav_file)

    # Read the generated .wav file and play it through speakers
    wav_data = np.fromfile("output.wav", dtype=np.int16)
    # Play the audio at the same sample rate as the .wav
    sd.play(wav_data, 22050)
    sd.wait()  # Wait until the audio finishes playing


ground_floor_hierarchical_tree = {
    "Department Entrance": {
        "In Front": {
            "Reception": "Central point connecting all branches",
            "Stairs": "Leads to the upper floor",
        },
        "Right Turn": {
            "Classroom 1": "1st room on the right side",
            "Industrial Automation Lab": "1st room on the left side",
            "Robotics Lab": "2nd room on the left side",
            "HOD Corridor": "on the Straight at the end of the corridor",
            "Secondary Exit": "on the Right side on end of the corridor",
        },
        "Left Turn": {
            "CAD/CAM Lab": "1st room on the right side",
            "Machine Vision Lab": "1st room on the left side",
            "Electronics Lab": "2nd room on the right side",
            "Washroom": "on the Left side on the end of the corridor",
            "Second stairs": "on the Left side on the end of the corridor",
        },
    }
}


def classify_input(sentence):
    sentence_transformed = loaded_vectorizer.transform([sentence])
    prediction = loaded_model.predict(sentence_transformed)
    print(f"The query '{sentence}' is classified as: {prediction[0]}")
    return prediction[0]


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
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    }

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # print(temp_history)
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-2024-05-13", messages=temp_history
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
    feel disabled. Scene Description: {agent_1_output}.
    """
    messages = {"role": "user", "content": prompt}
    messages_ = {"role": "user", "content": user_input}
    print("Content Prepared")

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # Call OpenAI API with image and text input, including conversation history
    conversation_history.append(messages_)
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=temp_history
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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ]
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-2024-05-13",
        response_format={"type": "json_object"},
        messages=messages,
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
    messages = [{"role": "user", "content": prompt}]
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages)
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
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt}]}]
    # print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages)

    # Get the AI's response content
    response_content = completion.choices[0].message.content
    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content


def ask_listener():
    print("Prompting user to speak...")
    converter.say("Say something")
    converter.runAndWait()


def wait_for_listen_command():
    while 1:
        ref = db.reference("/intValue").get()
        if ref == 1:
            break

    try:
        print("Initializing microphone...")
        # recognizer.adjust_for_ambient_noise(source, 2)  # Adjust for ambient
        # noise
        try:
            record_audio(AUDIO_FILE, SAMPLE_RATE)
            # Transcribe the recorded audio
            start_time = time.time()
            # result = model.transcribe(AUDIO_FILE)
            audio_file = open(AUDIO_FILE, "rb")
            transcription = openai.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            end_time = time.time()
            result = transcription.text

            # Print the transcribed text and processing time
            print(f"? Processing Time: {end_time - start_time:.2f} seconds")
            print("?? Transcribed Text:", result)
            return result

        except sr.UnknownValueError:
            # Handle case where speech was not understood
            print("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            # Handle API request issues
            print(
                f"Could not request results from Google Speech Recognition service; {e}")
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
    user_input = wait_for_listen_command()
    print(user_input)
    print("Command Captured")
    if True:
        classification_result = classify_input(user_input)
        if classification_result == "Scene Explanation":
            print("Capturing image...")
            img_path = get_image()
            base64_image = encode_image(img_path)
            print("Processing with Agent 1...")
            agent_1_output = explanation_agent_1(base64_image, user_input)
            print(f"Agent 1 Output: {agent_1_output}")
            print("Processing with Agent 2...")
            agent_2_output = explanation_agent_2(user_input, agent_1_output)
            print(f"Agent 2 Output: {agent_2_output}")
            convert_and_play_speech(agent_2_output)
        elif classification_result == "Local Navigation":
            print("Capturing image...")
            img_path = get_image()
            base64_image = encode_image(img_path)
            print("Processing with Agent 1...")
            agent_1_output = navigation_agent_1(base64_image, user_input)
            print(f"Agent 1 Output: {agent_1_output}")
            print("Processing with Agent 2...")
            agent_2_output = navigation_agent_2(user_input, agent_1_output)
            print(f"Agent 2 Output: {agent_2_output}")
            convert_and_play_speech(agent_2_output)
        else:
            print("Processing with Global Navigation Agent 1...")
            agent_1_output = global_navigation_agent(
                user_input, ground_floor_hierarchical_tree
            )
            initial_position, final_position = extract_positions_from_string(
                agent_1_output
            )
            print(
                f"Initial Position: {initial_position}                Final Position: {final_position}"
            )
            agent_2_output = generate_navigation(
                ground_floor_hierarchical_tree, initial_position, final_position)
            print(f"Agent 2 Output: {agent_2_output}")
            convert_and_play_speech(agent_2_output)
    elif user_input.lower() == "exit":
        print("Exiting the assistant.")
        break
    else:
        print("Say 'listen' to activate the assistant or say 'exit' to quit.")
