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
    speech_file_path = "output.wav"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Can you please describe my surrounding?Can you please describe my surrounding?Can you please describe my surrounding?Can you please describe my surrounding?Can you please describe my surrounding?Can you please describe my surrounding?",
    )
    response.stream_to_file(speech_file_path)
    t = t + time.time() - start_time

print(f"Total time taken: {t/10}")