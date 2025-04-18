import os
from os import listdir
from os.path import join, exists, splitext
import time
from picamera2 import Picamera2
from gpiozero import Button

# Define camera roles
CAMERA_LEFT = 'L'
CAMERA_RIGHT = 'R'

BASE_PATH = '/home/blindvision/STEREO_VISION'
PATH_L = join(BASE_PATH, CAMERA_LEFT + '_calib')
PATH_R = join(BASE_PATH, CAMERA_RIGHT + '_calib')

# Ensure directories exist
os.makedirs(PATH_L, exist_ok=True)
os.makedirs(PATH_R, exist_ok=True)

# Get current pair_id
existing_files = [f for f in listdir(PATH_L) if f.endswith(".jpg")]
if existing_files:
    fnames = [int(splitext(f)[0]) for f in existing_files]
    pair_id = max(fnames) + 1
else:
    pair_id = 0

# Simulate GPIO button for now
# btn = Button(24)  # GPIO pin if using hardware

try:
    # Initialize both cameras explicitly
    picam_L = Picamera2(1)  # cam0 = Left
    picam_R = Picamera2(0)  # cam1 = Right

    config_L = picam_L.create_still_configuration(main={"size": (1640, 1232)})
    config_R = picam_R.create_still_configuration(main={"size": (1640, 1232)})

    picam_L.configure(config_L)
    picam_R.configure(config_R)

    # Start both cameras
    picam_L.start()
    picam_R.start()
    time.sleep(2)  # Let them warm up

    # Wait for button press or simulate
    print("Waiting for button...")
    input("Press [Enter] to capture from both cameras...")  # Simulated button
    start = time.time()

    # File paths
    fname_L = join(PATH_L, f"{pair_id}.jpg")
    fname_R = join(PATH_R, f"{pair_id}.jpg")

    # Capture from both cameras
    picam_L.capture_file(fname_L)
    picam_R.capture_file(fname_R)

    finish = time.time()
    print(f"Captured pair_id {pair_id} in {finish - start:.3f} sec")
    print(f"Saved L: {fname_L}")
    print(f"Saved R: {fname_R}")

    # Cleanup
    picam_L.close()
    picam_R.close()

except Exception as e:
    print(f"\n>> Process Exception: {e}")
