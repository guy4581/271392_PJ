

import cv2
import os

# --------- Settings ---------
video_path = "VEDIO/IMG_6142.mp4"  # Input video file
interval = 2  # Capture every X seconds
# ----------------------------

# Create output directory if not exists
output_folder = "VEDIO/images"  # Output folder for images
os.makedirs(output_folder, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_interval = int(fps * interval)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame017_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print("Done.")
