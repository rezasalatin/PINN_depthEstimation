import cv2
import os
import glob

# Parameters
frame_rate = 5  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use the XVID codec
output_video_path = '../plots_irr/output_video.avi'  # Change file extension to .avi
image_folder = '../plots_irr'
image_files = sorted(glob.glob(f'{image_folder}/*.png'))

# Check if there are images to process
if not image_files:
    raise ValueError("No images found in the specified folder.")

# Determine the width and height from the first image
image_path = image_files[0]
frame = cv2.imread(image_path)
height, width, layers = frame.shape

# Video Writer
video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

for image in image_files:
    video.write(cv2.imread(image))

video.release()
cv2.destroyAllWindows()

print(f"Video saved to {output_video_path}")
