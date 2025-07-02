from openai import OpenAI
import time
import os
from dotenv import load_dotenv

if not os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key is missing. Please set it in the environment variable or directly in the script.")
else:
    print("Welcome! Type 'listen brother' to start a conversation.")

client = OpenAI(os.getenv("OPENAI_API_KEY"))

t = 0
for i in range(10):

    start_time = time.time()
    audio_file = open("recorded_audio.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    t = t + time.time() - start_time
    # print(time.time() - start_time)
    print(transcription.text)

print(f"Total time taken: {t / 10}")
