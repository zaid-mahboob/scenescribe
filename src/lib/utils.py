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
import noisereduce as nr
import scipy.io.wavfile as wavfile
import requests
import socket

cred_obj = firebase_admin.credentials.Certificate('credentials/credentials.json')
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL':'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
    })



text = ""

conversation_history = [{
    "role": "system",
    "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful."}]

video_prompt = "Please explain what is happening in the video!"
endpoint_url = "https://6e65-119-158-64-26.ngrok-free.app/analyze_video/"


user_text = None
lock = threading.Lock()  # To ensure thread-safe updates to `user_text`

model = whisper.load_model("tiny")
# Recording settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
AUDIO_FILE = "recorded_audio.wav"  # Output file
FRAME_DURATION = 30  # Frame size in milliseconds


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



class Utils:
    def __init__(self, picamera=None, recognizer=None, whisper_model=None, openai_client=None, shared_state: SharedState = None):
        self.latest_video_path = None
        self.picam2 = picamera if picamera else Picamera2()
        self.recognizer = recognizer if recognizer else sr.Recognizer()
        self.whisper_model = whisper_model if whisper_model else whisper.load_model("tiny")
        self.openai = openai_client
        self.shared_state = shared_state if shared_state else SharedState()

    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_image(self):
        image_path = "/home/scenescribe/Desktop/scenescribe/test.jpg"
        self.picam2.capture_file(image_path)
        return image_path

    def record_video_from_camera(self, duration=5, fps=2, output_filename="video_smolvlm.avi",
                                  resolution=(640, 640), video_dir="/home/scenescribe/Desktop/scenescribe/avis"):
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
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            time.sleep(1 / fps)

        # Cleanup
        out.release()
        print("✅ Recording finished.")
        self.latest_video_path = filepath
        return filepath

    def analyze_video_with_prompt(self,video_path = None, prompt_text=video_prompt, endpoint_url=endpoint_url):
        """
        Sends a video and prompt to an analysis API endpoint and returns the response.

        Args:
            video_path (str): Full path to the video file.
            prompt_text (str): Instruction or question for the model.
            endpoint_url (str): URL of the analysis API.

        Returns:
            dict: Response from the server containing status and text.
        """
        if video_path is None:
            video_path = self.latest_video_path
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
    
    def wait_for_listen_command(self):
        try:
                print("Initializing microphone...")
                try:
                    self.record_with_softap_control(AUDIO_FILE, SAMPLE_RATE, FRAME_DURATION)
                    # Transcribe the recorded audio
                    start_time = time.time()
                    # result = model.transcribe(AUDIO_FILE)
                    audio_file= open(AUDIO_FILE, "rb")
                    transcription = self.openai.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
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


    def convert_and_play_speech(self,  text):
        # Load the voice model
        model = "models/tts/en_GB-northern_english_male-medium.onnx"
        voice = PiperVoice.load(model)
        
        # Generate speech and save to .wav file
        with wave.open("output.wav", "w") as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
            wav_file.setframerate(22050)  # 22.05 kHz sample rate (adjust if needed)
            voice.synthesize(text, wav_file)
        
        # Read the generated .wav file and play it through speakers
        wav_data = np.fromfile("output.wav", dtype=np.int16)
        sd.play(wav_data, 22050)  # Play the audio at the same sample rate as the .wav
        sd.wait()  # Wait until the audio finishes playing

    def openai_convert_and_play_speech(self, text):
        # Call OpenAI TTS API
        response = self.openai.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice="nova",   # or "alloy", "echo", "fable", "nova", "shimmer"
            input=text
        )
        # Save the response audio content to a .wav file
        with open("output.wav", "wb") as f:
            f.write(response.content)
        
        # Read the generated .wav file and play it through speakers
        wav_data = np.fromfile("output.wav", dtype=np.int16)
        sd.play(wav_data, 22050)  # Play the audio at the same sample rate as the .wav
        sd.wait()  # Wait until the audio finishes playing

    def classify_input(self, sentence, loaded_model, loaded_vectorizer):
        sentence_transformed = loaded_vectorizer.transform([sentence])
        prediction = loaded_model.predict(sentence_transformed)
        print(f"The query '{sentence}' is classified as: {prediction[0]}")
        return prediction[0]


    def denoise_wav(self, file_path):
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
    def record_with_softap_control(self, output_path="recorded_audio.wav", sample_rate=16000, frame_duration=30):
        print("🔁 Waiting for Firebase button to turn ON...")

        # Wait for button ON
        while not self.shared_state.get_button_state():
            time.sleep(0.1)

        print("🎙️ Recording started...")
        buffer = []

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
            while self.shared_state.get_button_state():
                audio_frame, _ = stream.read(int(sample_rate * frame_duration / 1000))
                buffer.append(audio_frame)

        print("⏹️ Recording stopped. Saving...")

        audio_data = np.concatenate(buffer, axis=0)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        self.denoise_wav(output_path)

        print(f"✅ Finished. Audio saved and denoised at: {output_path}")

def check_network_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if the network connection is available.
    Tries to connect to a public DNS server (Google).
    Returns True if network is available, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

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
