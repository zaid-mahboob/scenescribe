#!/usr/bin/env python3
"""
SceneScribe - A visual assistance system for visually impaired individuals

This system uses computer vision, speech recognition, and AI to provide 
scene descriptions and navigation assistance to visually impaired users.
"""

import os
import sys
import time
import threading
import joblib
import firebase_admin
from firebase_admin import db
from openai import OpenAI
from picamera2 import Picamera2
import speech_recognition as sr
import whisper
from dotenv import load_dotenv
import signal
import logging
import sys
import time
# Import our modules
from ..lib.utils import SharedState, Utils, check_network_connection
from ..lib.agents import Agents

import logging
# Define the log file path
log_file = '/home/scenescribe/Desktop/scenescribe/error.log'

# Clear the log file if it exists
if os.path.exists(log_file):
    os.remove(log_file)

# Configure logging to capture errors
# logging.basicConfig(filename=log_file, level=logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # <-- prints to terminal
    ]
)


try:
    # Your script logic here
    logging.info("Starting the script")
    
    # Example script code that might raise an error
    # result = 1 / 0  # Example of an error (divide by zero)
    
except Exception as e:
    logging.error(f"Error occurred: {e}")
    raise

# # Define a signal handler to ignore SIGINT (Ctrl+C)
# def signal_handler(sig, frame):
#     logging.info("Ctrl+C was pressed, but the script will continue running.")
#     # Do nothing, just return so the script keeps running
#     pass

# # Attach the signal handler to SIGINT
# signal.signal(signal.SIGINT, signal_handler)

class SceneScribe:
    def __init__(self, shared_state: SharedState, language="Urdu"):
        """
        Initialize the SceneScribe application.
        
        Args:
            language: Language for output (defaults to English)
        """
        logging.info(f"Initializing SceneScribe with {language} language...")
        self.sharedState = shared_state
        self.language = language
        self.setup_environment()
        self.load_models()
        self.initialize_camera()
        # self.initialize_firebase()
        
        # Initialize conversation history
        self.conversation_history = [{
            "role": "system",
            "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful."
        }]
        
        # Initialize utility functions and agents
        self.utils = Utils(
            picamera=self.picamera,
            recognizer=self.recognizer,
            whisper_model=self.whisper_model,
            openai_client=self.openai,
            shared_state = self.sharedState, 
        )
        
        self.agents = Agents(
            openai_client=self.openai, 
            conversation_history=self.conversation_history,
            language=self.language
        )
        
        # API endpoint for video analysis
        self.endpoint_url = "https://ccdd-59-103-82-46.ngrok-free.app/analyze_video/"
        self.educational_mode = False
        
        
    def setup_environment(self):
        """Set up environment variables and suppress warnings"""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment or set directly
        self.openai_key = os.getenv("OPENAI_API_KEY") 
        
        if not self.openai_key:
            logging.info("OpenAI API key is missing. Please set it in the environment variable or directly in the script.")
            sys.exit(1)
            
        # Suppress standard error for cleaner output
        sys.stderr = open(os.devnull, 'w')
    
    def load_models(self):
        """Load ML models and initialize clients"""
        logging.info("Loading models...")
        
        # Initialize OpenAI client
        self.openai = OpenAI(api_key=self.openai_key)
        
        # Load classification model and vectorizer
        self.loaded_model = joblib.load("/home/scenescribe/Desktop/scenescribe/models/nb_classifier_3_classes_v2.pkl")
        self.loaded_vectorizer = joblib.load("/home/scenescribe/Desktop/scenescribe/models/vectorizer_3_classes_v2.pkl")
        
        # Initialize speech recognition components
        self.recognizer = sr.Recognizer()
        self.whisper_model = whisper.load_model("tiny")
        
        logging.info("Models loaded successfully")
    
    def initialize_camera(self):
        """Initialize PiCamera2 for capturing images and video"""
        logging.info("Initializing Camera...")
        
        self.picamera = Picamera2()
        # camera_config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
        # picam2.configure(camera_config)
        self.picamera.start()
        time.sleep(0.1)
        # picam2.set_controls({"AfMode": 2})
        # picam2.set_controls({"AfTrigger": 0})
        
        logging.info("Camera Initialized")
    
    def initialize_firebase(self):
        """Initialize Firebase for real-time database access"""
        logging.info("Initializing Firebase...")
        
        cred_obj = firebase_admin.credentials.Certificate('/home/scenescribe/Desktop/scenescribe/credentials/credentials.json')
        firebase_admin.initialize_app(cred_obj, {
            'databaseURL': 'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        
        logging.info("Firebase Initialized")
    
    def process_scene_explanation(self, user_input):
        """
        Handle scene explanation queries by capturing an image and describing it.
        
        Args:
            user_input: User's query text
            
        Returns:
            str: Scene description
        """
        logging.info("Capturing image for scene explanation...")
        img_path = self.utils.get_image()
        base64_image = self.utils.encode_image(img_path)
        
        logging.info("Processing with Agent 1...")
        agent_1_output = self.agents.explanation_agent_1(base64_image, user_input)
        logging.info(f"Agent 1 Output: {agent_1_output}")
        
        logging.info("Processing with Agent 2...")
        agent_2_output = self.agents.explanation_agent_2(user_input, agent_1_output)
        logging.info(f"Agent 2 Output: {agent_2_output}")
        
        return agent_2_output
    
    def process_educational_explanation(self, user_input):
        """
        Handle educational explanation queries by capturing an image and describing it.
        
        Args:
            user_input: User's query text
            
        Returns:
            str: educational description
        """
        logging.info("Capturing image for educational explanation...")
        img_path = self.utils.get_image()
        base64_image = self.utils.encode_image(img_path)
        
        logging.info("Processing with Agent 1...")
        agent_1_output = self.agents.educational_agent_1(base64_image, user_input)
        logging.info(f"Agent 1 Output: {agent_1_output}")
        
        logging.info("Processing with Agent 2...")
        agent_2_output = self.agents.educational_agent_2(user_input, agent_1_output)
        logging.info(f"Agent 2 Output: {agent_2_output}")
        
        return agent_2_output
    
    def process_navigation(self, user_input):
        """
        Handle navigation queries by capturing an image and providing directions.
        
        Args:
            user_input: User's query text
            
        Returns:
            str: Navigation instructions
        """
        logging.info("Capturing image for navigation...")
        img_path = self.utils.get_image()
        base64_image = self.utils.encode_image(img_path)
        
        logging.info("Processing with Navigation Agent 1...")
        agent_1_output = self.agents.navigation_agent_1(base64_image, user_input)
        logging.info(f"Navigation Agent 1 Output: {agent_1_output}")
        
        logging.info("Processing with Navigation Agent 2...")
        agent_2_output = self.agents.navigation_agent_2(user_input, agent_1_output)
        logging.info(f"Navigation Agent 2 Output: {agent_2_output}")
        
        return agent_2_output
    
    def process_video_analysis(self, user_input):
        """
        Handle detailed scene analysis using video recording.
        
        Args:
            user_input: User's query text
            
        Returns:
            str: Detailed scene description based on video
        """
        logging.info("Using Video Analysis...")
        output = self.agents.activity_detection(
            user_input, 
            self.utils.record_video_from_camera,
            self.utils.analyze_video_with_prompt,
            self.endpoint_url
        )
        return output
    
    def run(self):
        """
        Main application loop that listens for commands and processes them.
        """
        logging.info("Welcome to SceneScribe! Ready to listen for commands.")
        
        while True:
            try:
                # Wait for user input via voice
                logging.info("Waiting for voice command...")
                user_input = self.utils.wait_for_listen_command()
                
                if not user_input:
                    logging.info("No input detected, trying again...")
                    continue
                
                logging.info(f"Command received: {user_input}")
                start_time  = time.time()
                # Exit condition
                if user_input.lower() in ["exit", "quit", "goodbye"]:
                    logging.info("Exiting SceneScribe. Goodbye!")
                    break
                if "education" in user_input.lower():
                    self.educational_mode = not self.educational_mode
                    if self.educational_mode:
                        self.utils.openai_convert_and_play_speech("Education Mode Enabled")
                    else:
                        self.utils.openai_convert_and_play_speech("Education Mode Disabled")
                    print(f"Educational Mode set to: {self.educational_mode}")
                    continue
                
                if self.educational_mode:
                    print("Using Educational Mode")
                    output = self.process_educational_explanation(user_input)
                # Process based on input classification
                # elif "detail" in user_input.lower():
                #     # For detailed analysis, use video processing
                #     output = self.process_video_analysis(user_input)
                else:
                    if not check_network_connection():
                        output = "Sorry not connected to internet" 
                    else:
                        # Classify the input to determine processing approach
                        classification = self.utils.classify_input(
                            user_input, 
                            self.loaded_model, 
                            self.loaded_vectorizer
                        )
                        
                        if classification == "Scene Explanation":
                            output = self.process_scene_explanation(user_input)
                        else:
                            output = self.process_navigation(user_input)
                end_time = time.time()
                print(f"TIme Taken: {end_time - start_time}")
                # Convert text response to speech
                print(f"Output: {output}")
                if output:
                    self.utils.openai_convert_and_play_speech(output)
                print(f"Output time: {time.time() - end_time}")
            except KeyboardInterrupt:
                logging.info("\nProgram interrupted by user. Exiting...")
                break
            except Exception as e:
                logging.info(f"Error in main loop: {e}")
                # Try to communicate error to user
                error_msg = f"I'm sorry, I encountered an error: {str(e)}"
                try:
                    self.utils.openai_convert_and_play_speech(error_msg)
                except:
                    pass


# def main():
#     """
#     Entry point for the SceneScribe application.
#     """
#     # Use language from command line arg if provided, otherwise default to English
#     language = sys.argv[1] if len(sys.argv) > 1 else "English"
    
#     # Create and run SceneScribe instance
#     app = SceneScribe(language=language)
#     app.run()

