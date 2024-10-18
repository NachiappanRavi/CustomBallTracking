import os
from ultralytics import YOLO
import torch
import cv2
import json
import numpy as np
import logging

# Set up the output directory
output_dir = r"output_videos_custom_yolo"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
log_file_path = os.path.join(output_dir, "ball_coordinates.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')

# Load your custom YOLOv8 model (replace with your custom-trained model path)
model = YOLO(r"C:\Local_Disk_D\Git\CustomBallTracking\custom_ball_dataset.pt")  # Replace 'path/to/custom_best.pt' with your actual model path
# model = YOLO(r'yolov10x.pt')
# Path to input and output videos
input_video_path = r"C:\Local_Disk_D\Git\CustomBallTracking\Input_Video\IMG_1161.mp4"
output_video_path = os.path.join(output_dir, "output_video.avi")
coordinates_json_path = os.path.join(output_dir, "coordinates.json")

# Run the model prediction on the input video
results = model.predict(input_video_path, save=False)

# Open the input video using OpenCV
cap = cv2.VideoCapture(input_video_path)

# Get video properties (width, height, frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# List to store the center coordinates
trajectory_coords = []

# Iterate over each frame and detection result
for frame_idx, result in enumerate(results):
    # Read the corresponding frame from the input video
    ret, frame = cap.read()
    if not ret:
        break

    # Create a temporary list to store trajectory for the current frame
    frame_coords = []

    # Iterate over each detection in the frame
    for box in result.boxes:
        # Check if the detected object is your custom class for cricket ball (adjust this class ID if needed)
        if int(box.cls[0]) ==  0:  # Replace YOUR_CLASS_ID with the correct class ID for cricket ball in your custom dataset
            # Get the bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Calculate the center coordinates of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get the confidence score for the detected object
            confidence = box.conf[0].item()

            # Adjusted confidence threshold
            if confidence < 0.5:  # Increase or adjust this threshold as needed
                continue
            
            # Add the center coordinates to the frame's trajectory list
            frame_coords.append((center_x, center_y))

            # Draw the bounding box on the frame  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the center point on the frame
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue dot for center

            # Display the center coordinates on the frame
            cv2.putText(frame, f'Center: ({center_x}, {center_y})', (center_x + 10, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            print(f'Frame {frame_idx}: Ball center at ({center_x}, {center_y})')

            # Log the center coordinates
            log_message = f'Frame {frame_idx}: Ball center at ({center_x}, {center_y})'
            logging.info(log_message)

    # If a ball was detected in this frame, update the global trajectory list
    if frame_coords:
        trajectory_coords.extend(frame_coords)
    # else:
    #     # Clear the trajectory if no ball is detected
    #     trajectory_coords.clear()

    # Draw a stable trajectory line with an invisible effect for older points
    if len(trajectory_coords) > 1:
        overlay = frame.copy()
        
        # Loop over trajectory points to draw segments with decreasing alpha (transparency)
        num_points = len(trajectory_coords)
        max_opacity = 255  # Maximum opacity
        fade_length = min(10, num_points)  # Length of the fading effect
        
        # Draw segments of the trajectory line
        for i in range(1, num_points):
            # Set opacity to fade older points
            opacity = int(max_opacity * (fade_length - (num_points - i)) / fade_length)
            opacity = max(0, opacity)  # Ensure opacity is non-negative
            
            # Set the color with transparency (use green with variable opacity)
            color = (0, 255, 0, opacity)  # Green color with fading effect
            
            # Draw the segment between point i-1 and i
            cv2.line(overlay, trajectory_coords[i - 1], trajectory_coords[i], color[:3], 12)
        
        # Merge the overlay with the original frame
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Save the frame to the output video
    out.write(frame)

# Release video capture and writer objects
cap.release()
out.release()

# Save the trajectory coordinates to a JSON file
# with open(coordinates_json_path, 'w') as json_file:
#     json.dump(trajectory_coords, json_file)

print(f"Processed video saved at {output_video_path}")
print(f"Coordinates saved at {coordinates_json_path}")
print(f"Ball center coordinates logged at {log_file_path}")

