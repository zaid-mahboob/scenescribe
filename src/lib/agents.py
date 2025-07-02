from openai import OpenAI
import logging
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
import noisereduce as nr
import scipy.io.wavfile as wavfile
import requests


conversation_history = [
    {
        "role": "system",
        "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful.",
    }
]

video_prompt = "Please explain what is happening in the video!"
endpoint_url = "https://6e65-119-158-64-26.ngrok-free.app/analyze_video/"


class Agents:
    def __init__(self, openai_client=None, conversation_history=None, language="Urdu"):
        self.openai = openai_client
        if self.openai is None:
            logging.error("OpenAI client is not initialized.")
        self.conversation_history = conversation_history
        self.language = language

    def explanation_agent_1(self, image_base64, user_input):
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
                {"type": "text", "text": prompt},
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
        # Call OpenAI API with image and text input, including conversation
        # history
        completion = self.openai.chat.completions.create(
            model="gpt-4o-2024-05-13", messages=temp_history
        )

        # Get the AI's response content
        response_content = completion.choices[0].message.content

        # converter.say(response_content)
        # converter.runAndWait()
        # print(response_content)
        return response_content

    def explanation_agent_2(self, user_input, agent_1_output):
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
        # Call OpenAI API with image and text input, including conversation
        # history
        conversation_history.append(messages_)
        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=temp_history
        )
        output = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": output})
        # print(conversation_history)

        # converter.say(output)
        # converter.runAndWait()
        return output

    def navigation_agent_1(self, image_base64, user_input):
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
        # Call OpenAI API with image and text input, including conversation
        # history
        completion = self.openai.chat.completions.create(
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

    def navigation_agent_2(self, user_input, agent_1_output):
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
        # Call OpenAI API with image and text input, including conversation
        # history
        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        output = completion.choices[0].message.content
        # converter.say(output)
        # converter.runAndWait()
        return output

    def global_navigation_agent(self, user_input, tree):
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
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        # print("Content Prepared")
        # Call OpenAI API with image and text input, including conversation
        # history
        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )

        # Get the AI's response content
        response_content = completion.choices[0].message.content
        # converter.say(response_content)
        # converter.runAndWait()
        # print(response_content)
        return response_content

    def analyze_video_with_prompt(self, video_path, prompt_text, endpoint_url):
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
            with open(video_path, "rb") as video_file:
                files = {"video": video_file}
                data = {"prompt": prompt_text}

                response = requests.post(endpoint_url, files=files, data=data)

                return {
                    "status_code": response.status_code,
                    "response_text": response.text,
                }

        except Exception as e:
            return {"status_code": None, "response_text": f"Error: {e}"}

    def activity_detection(self, user_input, video_path, response, endpoint_url):

        response = self.analyze_video_with_prompt(
            video_path=video_path, prompt_text=video_prompt, endpoint_url=endpoint_url
        )
        full_response = response["response_text"]

        # Extract text after 'assistant:'
        if "assistant:" in full_response.lower():
            # Case-insensitive search
            assistant_output = full_response.lower().split("assistant:", 1)[-1].strip()
            print("🧠 Assistant said:", assistant_output)
            return assistant_output
        return False
