import wave
import sounddevice as sd
import numpy as np
from piper.voice import PiperVoice
import time


def convert_and_play_speech(text):
    # Load the voice model
    model = "tts/en_GB-northern_english_male-medium.onnx"
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


# Example usage:
text = "Can you please describe my surrounding?"

start_time = time.time()
convert_and_play_speech(text)
print(time.time() - start_time)
print("Here we go boss")
