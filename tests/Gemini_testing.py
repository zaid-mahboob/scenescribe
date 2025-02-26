import google.generativeai as genai
import PIL.Image
import time
import os 
import pyttsx3 

converter = pyttsx3.init()

start_time = time.time()
os.system("libcamera-jpeg -o z.jpg")
print("Saved")
img = PIL.Image.open("z.jpg")

genai.configure(api_key='AIzaSyD3vfEy0ePzbqyov3_OBt7nt2BnNewiD8Y')
model = genai.GenerativeModel('gemini-pro-vision')

# response = model.generate_content(["Suppose you are an assistant for a visually impaired, act as an assistant and guide him in road navigation. Tell in such a manner that it gives a concise surrounding perception and help him navigate on roads. Keep it small and concise", img], stream=True)
response = model.generate_content(["Suppose you are an assistant for a visually impaired, act as an assistant and guide him in indoor environment. Tell in such a manner that it gives a concise surrounding perception and give him overview of hi surroundings. Keep it small and concise", img], stream=True)
response.resolve()

print(f"Time taken is: {time.time() - start_time:.2f}")

converter.say(response.text)
converter.runAndWait()

print(response.text)

