import whisper
import time
import sounddevice as sd
import numpy as np
import wave
import webrtcvad

# Load Whisper model (Use "tiny" or "base" for Raspberry Pi)
model = whisper.load_model("tiny")

# Recording settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
AUDIO_FILE = "recorded_audio.wav"  # Output file
FRAME_DURATION = 30  # Frame size in milliseconds
VAD = webrtcvad.Vad(3)  # Aggressiveness (0-3): 3 is most sensitive


def is_speech(audio_frame):
    """Check if the frame contains speech."""
    return VAD.is_speech(audio_frame.tobytes(), SAMPLE_RATE)


def record_audio(file_path, sample_rate):
    """Records audio from the microphone until silence is detected."""
    print("?? Listening... Speak now!")

    buffer = []
    silence_counter = 0
    max_silence_frames = 50  # Stop recording after 10 silent frames (~0.3 sec)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
        while True:
            audio_frame, _ = stream.read(
                int(sample_rate * FRAME_DURATION / 1000))  # Read a small chunk
            buffer.append(audio_frame)

            if is_speech(audio_frame):
                silence_counter = 0  # Reset silence counter if speech is detected
            else:
                silence_counter += 1  # Increment silence counter

            if silence_counter > max_silence_frames:  # Stop if silence persists
                break

    print("? Speech detected. Processing...")

    # Convert buffer to NumPy array
    audio_data = np.concatenate(buffer, axis=0)

    # Save to WAV file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


# Adaptive recording
record_audio(AUDIO_FILE, SAMPLE_RATE)

# Transcribe the recorded audio
start_time = time.time()
result = model.transcribe(AUDIO_FILE)
end_time = time.time()

# Print the transcribed text and processing time
print(f"? Processing Time: {end_time - start_time:.2f} seconds")
print("?? Transcribed Text:", result["text"])
