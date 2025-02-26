import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# List available microphones
microphones = sr.Microphone.list_microphone_names()
for idx, name in enumerate(microphones):
    print(f"{idx}: {name}")

# # Choose a microphone index (e.g., 1 for USB Microphone)
MIC_INDEX = 0  # Replace with the desired microphone index

# Use the chosen microphone
with sr.Microphone(device_index=MIC_INDEX) as source:
    print("Say something:")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
