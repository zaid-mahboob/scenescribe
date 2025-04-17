import whisper
import time
import sounddevice as sd
import numpy as np
import wave
import firebase_admin
from firebase_admin import db
import threading
import time

# --- Setup Whisper Model ---
model = whisper.load_model("tiny")

# --- Recording Settings ---
SAMPLE_RATE = 16000
AUDIO_FILE = "recorded_audio.wav"
FRAME_DURATION = 30  # ms

# --- Firebase Setup ---
cred_obj = firebase_admin.credentials.Certificate('/home/scenescribe/Desktop/scenescribe/credentials/credentials.json')
firebase_admin.initialize_app(cred_obj, {
    'databaseURL': 'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# --- Shared State ---
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
        time.sleep(0.1)  # Poll every 100ms

# --- Audio Recording Logic ---
def record_audio(file_path, sample_rate, shared: SharedState):
    print("🔴 Waiting for Firebase button to turn ON...")

    # Wait until the button is turned ON
    while not shared.get_button_state():
        time.sleep(0.1)

    print("🎙️ Recording started...")
    start_time = time.time()
    buffer = []

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
        while shared.get_button_state():
            audio_frame, _ = stream.read(int(sample_rate * FRAME_DURATION / 1000))
            buffer.append(audio_frame)
            print(len(buffer))

    print("⏹️ Recording stopped. Saving...")
    print(f"Recording Duration Should be: {time.time() - start_time}")

    # Combine and save
    audio_data = np.concatenate(buffer, axis=0)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

# --- Launch ---
if __name__ == "__main__":
    # Start Firebase polling in a background thread
    firebase_thread = threading.Thread(target=firebase_polling_loop, args=(shared_state,), daemon=True)
    firebase_thread.start()

    # Start the audio recording loop
    record_audio(AUDIO_FILE, SAMPLE_RATE, shared_state)
