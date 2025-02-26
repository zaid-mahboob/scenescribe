import os
import time
start_time = time.time()
os.system(f"libcamera-still -t 5000 --autofocus-mode auto -o autofocus_test{2}.jpg")
print(f"image taking time: {time.time() -start_time}")