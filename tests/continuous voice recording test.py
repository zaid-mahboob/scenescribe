import speech_recognition as sr
import time
import firebase_admin
from firebase_admin import db

cred_obj = firebase_admin.credentials.Certificate('/home/scenescribe/Desktop/scenescribe/credentials.json')
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL':'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
    })


# Initialize recognizer
recognizer = sr.Recognizer()

# Store the accumulated audio
accumulated_audio = []

def wait_for_listen_command():
    while True:
        # Get the value of intValue from Firebase
        ref = db.reference("/intValue").get()
        
        # Start listening when intValue is 1
        if ref == 1:
            print("Button is pressed (intValue = 1). Starting recording...")
            break
        time.sleep(1)  # Wait for a while before checking again
    
    try:
        with sr.Microphone(device_index=0) as source:
            print("Initializing microphone...")
            recognizer.adjust_for_ambient_noise(source, 2)  # Adjust for ambient noise

            while True:
                ref = db.reference("/intValue").get()
                
                if ref == 1:
                    try:
                        # Continuously listen and accumulate audio while the button is pressed
                        audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
                        print("Audio captured and accumulated...")
                        accumulated_audio.append(audio)  # Store the audio segments
                        
                    except sr.UnknownValueError:
                        print("Could not understand the audio. Please try again.")
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
                else:
                    print("Button released (intValue != 1). Stopping recording...")
                    break
                
                time.sleep(0.1)  # Prevent busy-waiting, adjust if needed

        # Once the button is released, process the accumulated audio
        if accumulated_audio:
            print("Button released. Sending accumulated audio to Google Speech API...")

            # Join the accumulated audio and create a single AudioData object
            combined_audio = b''.join([segment.get_wav_data() for segment in accumulated_audio])

            # Convert the combined audio into an AudioData object with the same sample rate and sample width as the original
            full_audio = sr.AudioData(combined_audio, accumulated_audio[0].sample_rate, accumulated_audio[0].sample_width)

            try:
                # Pass the full accumulated audio to Google Speech API for recognition
                text = recognizer.recognize_google(full_audio)
                print(f"You said: {text}")
                # You can pass the text to another system if needed

            except sr.UnknownValueError:
                print("Could not understand the accumulated audio.")
            except sr.RequestError as e:
                print(f"Error with Speech Recognition service: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
while(1):
    wait_for_listen_command()