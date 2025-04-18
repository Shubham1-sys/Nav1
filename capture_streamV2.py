import os
import time
from picamera2 import Picamera2
from os.path import join

# Constants
BASE_PATH = '/home/blindvision/STEREO_VISION'
CAM_L_PATH = os.path.join(BASE_PATH, 'L_stream')
CAM_R_PATH = os.path.join(BASE_PATH, 'R_stream')
RECORD_SECONDS = 10
FPS = 10
FRAME_INTERVAL = 1 / FPS
RESOLUTION = (1640, 1232)

# Ensure directories exist
os.makedirs(CAM_L_PATH, exist_ok=True)
os.makedirs(CAM_R_PATH, exist_ok=True)

try:
    print("Initializing cameras...")
    camL = Picamera2(1)
    camR = Picamera2(0)

    camL.configure(camL.create_still_configuration(main={"size": RESOLUTION}))
    camR.configure(camR.create_still_configuration(main={"size": RESOLUTION}))

    camL.start()
    camR.start()
    time.sleep(2)  # Warm-up

    input("Press [Enter] to start synchronized capture...")

    total_frames = int(RECORD_SECONDS * FPS)
    print(f"Capturing {total_frames} frames from both cameras...")

    for frame_num in range(total_frames):
        start_time = time.time()

        fname_L = join(CAM_L_PATH, f"{frame_num}.jpg")
        fname_R = join(CAM_R_PATH, f"{frame_num}.jpg")

        camL.capture_file(fname_L)
        camR.capture_file(fname_R)

        print(f"Captured frame {frame_num}")

        elapsed = time.time() - start_time
        wait = FRAME_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)

    camL.close()
    camR.close()

    print(f"✅ Done! {total_frames} frames captured for both cameras.")

except Exception as e:
    print(f"❌ Exception: {e}")
