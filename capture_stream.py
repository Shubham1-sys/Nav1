import os
from os import listdir, mkdir
from os.path import join, exists, splitext
import time
from picamera2 import Picamera2

# Define camera roles and paths
CAMERA_LEFT = 'L'
CAMERA_RIGHT = 'R'
BASE_PATH = '/home/blindvision/STEREO_VISION'
PATH_L = join(BASE_PATH, 'L_stream')  # Left camera stream directory
PATH_R = join(BASE_PATH, 'R_stream')  # Right camera stream directory

# Ensure directories exist
os.makedirs(PATH_L, exist_ok=True)
os.makedirs(PATH_R, exist_ok=True)

def get_next_frame_id():
    """Get the next sequential frame ID from existing files"""
    existing_files = [f for f in listdir(PATH_L) if f.endswith(".jpg")]
    if not existing_files:
        return 0
    return max([int(splitext(f)[0]) for f in existing_files]) + 1

def capture_stream():
    # Initialize cameras
    picam_L = Picamera2(1)  # Left camera (cam0)
    picam_R = Picamera2(0)  # Right camera (cam1)

    # Configure both cameras
    config = picam_L.create_still_configuration(main={"size": (1640, 1232)})
    picam_L.configure(config)
    picam_R.configure(config)

    # Start cameras
    picam_L.start()
    picam_R.start()
    time.sleep(2)  # Warm-up

    frame_id = get_next_frame_id()

    try:
        while True:
            input("Press [Enter] to capture a frame (Ctrl+C to exit)...")
            
            # Capture synchronized frames
            fname_L = join(PATH_L, f"{frame_id}.jpg")
            fname_R = join(PATH_R, f"{frame_id}.jpg")

            picam_L.capture_file(fname_L)
            picam_R.capture_file(fname_R)
            print(f"Captured pair: {frame_id}.jpg")

            frame_id += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        picam_L.stop()
        picam_R.close()
        picam_L.close()

if __name__ == "__main__":
    capture_stream()