# crop video

import cv2
from pathlib import Path

input_video = 'videos/d14.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video)

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4

# Initialize the video writer with the NEW cropped dimensions (w, h)
left_out = cv2.VideoWriter("videos/" + Path(input_video).stem + "_left.mp4", fourcc, fps, (960, 540))
right_out = cv2.VideoWriter("videos/" + Path(input_video).stem + "_right.mp4", fourcc, fps, (960, 540))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id = 0

while True:
    ret, frame = cap.read()
    
    # If no frame is returned, the video has ended
    if not ret:
        break

    # Crop the frame using NumPy slicing: [ymin:ymax, xmin:xmax]
    cropped_frame_r = frame[0:540, 0:960]
    cropped_frame_l = frame[0:540, 960:1920]

    # Write the cropped frame to the output file
    left_out.write(cropped_frame_l)
    right_out.write(cropped_frame_r)
    
    # Progress indicator
    frame_id += 1
    pct = frame_id / frame_count * 100
    print(f"\rProgress: {pct:.1f}%", end="")

# Release everything when done
cap.release()
left_out.release()
right_out.release()
cv2.destroyAllWindows()


