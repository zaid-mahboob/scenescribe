import pyttsx3 
import time
converter = pyttsx3.init()

text = "There is a wide road in front of you with a pedestrian crossing It is a four-lane road with a wide median in the middle. The median is planted with trees and grass. There are cars and buses on the road, and people are crossing the street. The traffic lights are green for cars and red for pedestrians."
text2 = "Your name is. Zaid Mahboob."
text3 = text2 + ". ."
print(text3)
converter.setProperty('rate', 150)
segments = text.split('.')
for segment in segments:
    if segment.strip():
        converter.say(segment.strip() + '.')  # Add the full stop back for clarity
        converter.runAndWait()
        time.sleep(0.5)  # Short pause between segments
# converter.say(text)
# converter.runAndWait()

print("hejfjf jsdjs")