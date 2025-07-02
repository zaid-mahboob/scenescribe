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
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import db

cred_obj = firebase_admin.credentials.Certificate(
    "/home/scenescribe/Desktop/scenescribe/credentials.json"
)
default_app = firebase_admin.initialize_app(
    cred_obj,
    {
        "databaseURL": "https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app"
    },
)

# Load the saved model and vectorizer
loaded_model = joblib.load("models/nb_classifier_3_classes.pkl")
loaded_vectorizer = joblib.load("models/vectorizer_3_classes.pkl")

# Initialize the recognizer
sys.stderr = open(os.devnull, "w")
converter = pyttsx3.init()
recognizer = sr.Recognizer()

picam2 = Picamera2()
picam2.start()

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

openai = OpenAI(os.getenv("OPENAI_API_KEY"))
user_text = None
lock = threading.Lock()  # To ensure thread-safe updates to `user_text`


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
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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
        with sr.Microphone(device_index=0) as source:
            print("Initializing microphone...")
            recognizer.adjust_for_ambient_noise(
                source, 2)  # Adjust for ambient noise
            try:
                print("You can say now anything")
                # Capture audio
                audio = recognizer.listen(
                    source, timeout=5, phrase_time_limit=60)

                # Recognize speech using Google API
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text

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
    print("Command Captured")
    if True:
        classification_result = classify_input(user_input)
        if classification_result == "Explanation":
            print("Capturing image...")
            img_path = get_image()
            base64_image = encode_image(img_path)
            print("Processing with Agent 1...")
            agent_1_output = explanation_agent_1(base64_image, user_input)
            print(f"Agent 1 Output: {agent_1_output}")
            print("Processing with Agent 2...")
            agent_2_output = explanation_agent_2(user_input, agent_1_output)
            print(f"Agent 2 Output: {agent_2_output}")
            converter.setProperty("rate", 150)
            agent_2_output = agent_2_output + ". ."
            segments = agent_2_output.split(".")
            for segment in segments:
                if segment.strip():
                    converter.say(
                        segment.strip() + "."
                    )  # Add the full stop back for clarity
                    converter.runAndWait()
                    time.sleep(0.5)  # Short pause between segments
        elif classification_result == "Navigation":
            print("Capturing image...")
            img_path = get_image()
            base64_image = encode_image(img_path)
            print("Processing with Agent 1...")
            agent_1_output = navigation_agent_1(base64_image, user_input)
            print(f"Agent 1 Output: {agent_1_output}")
            print("Processing with Agent 2...")
            agent_2_output = navigation_agent_2(user_input, agent_1_output)
            print(f"Agent 2 Output: {agent_2_output}")
            converter.setProperty("rate", 150)
            agent_2_output = agent_2_output + ". ."
            segments = agent_2_output.split(".")
            for segment in segments:
                if segment.strip():
                    converter.say(
                        segment.strip() + "."
                    )  # Add the full stop back for clarity
                    converter.runAndWait()
                    time.sleep(0.5)  # Short pause between segments
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
            converter.say(agent_2_output)
            converter.runAndWait()
    elif user_input.lower() == "exit":
        print("Exiting the assistant.")
        break
    else:
        print("Say 'listen' to activate the assistant or say 'exit' to quit.")
