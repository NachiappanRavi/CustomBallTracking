import os
import cv2
import numpy as np
from ultralytics import YOLO


# Load the YOLO model
def load_model(model_path):
    return YOLO(model_path)


# Load video and set up writer
def load_video(video_path, desired_width, desired_height, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (desired_width, desired_height))
    return cap, video_writer, fps


# Manual ground selection
def select_ground_area(frame, desired_width, desired_height):
    ground_selection = cv2.selectROI("Select Ground Area", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Ground Area")
    mask = np.zeros((desired_height, desired_width), dtype=np.uint8)
    if ground_selection:
        x, y, w, h = ground_selection
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # White rectangle for the ground area
    return mask


# Function to calculate angle and direction of deviation
def calculate_deviation(stump_line, ball_position):
    stump_vector = np.array(stump_line[1]) - np.array(stump_line[0])
    ball_vector = np.array(ball_position) - np.array(stump_line[0])
    stump_vector = stump_vector / np.linalg.norm(stump_vector)
    ball_vector = ball_vector / np.linalg.norm(ball_vector)

    dot_product = np.dot(stump_vector, ball_vector)
    angle = np.arccos(dot_product) * (180.0 / np.pi)

    cross_product = np.cross(stump_vector, ball_vector)
    direction = "Right" if cross_product > 0 else "Left"
    return angle, direction


# Process each frame to detect the ball and calculate deviation
def process_frame(frame, mask, model, stump_boxes, trajectory_coords, bounce_coords_path, cap):
    frame_copy = frame.copy()
    results = model.predict(frame_copy, save=False)
    angle, direction = None, None

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Assuming class ID for the cricket ball is 0
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                trajectory_coords.append((center_x, center_y))

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame_copy, (center_x, center_y), 5, (0, 0, 255), -1)

                if mask is not None and mask[y1:y2, x1:x2].any():
                    if len(stump_boxes) == 2:
                        x1_center = int(stump_boxes[0][0] + stump_boxes[0][2] // 2)
                        y1_bottom = int(stump_boxes[0][1] + stump_boxes[0][3])
                        x2_center = int(stump_boxes[1][0] + stump_boxes[1][2] // 2)
                        y2_bottom = int(stump_boxes[1][1] + stump_boxes[1][3])

                        stump_line = [(x1_center, y1_bottom), (x2_center, y2_bottom)]
                        ball_position = (center_x, center_y)

                        angle, direction = calculate_deviation(stump_line, ball_position)

                        with open(bounce_coords_path, 'a') as f:
                            f.write(
                                f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, Coords: ({center_x}, {center_y}), Angle: {angle:.2f}, Direction: {direction}\n")

    return frame_copy, angle, direction


# Draw the trajectory of the ball
def draw_trajectory(frame, trajectory_coords):
    if len(trajectory_coords) > 1:
        for i in range(1, len(trajectory_coords)):
            cv2.line(frame, trajectory_coords[i - 1], trajectory_coords[i], (255, 0, 0), 2)


# Draw the stump line
def draw_stump_line(frame, stump_boxes):
    if len(stump_boxes) == 2:
        # Create a copy of the frame for overlay
        overlay = frame.copy()

        # Get the center and bottom of each stump box
        x1_center = int(stump_boxes[0][0] + stump_boxes[0][2] // 2)
        y1_bottom = int(stump_boxes[0][1] + stump_boxes[0][3])
        x2_center = int(stump_boxes[1][0] + stump_boxes[1][2] // 2)
        y2_bottom = int(stump_boxes[1][1] + stump_boxes[1][3])

        # Define the color (light purple) and thickness
        light_purple = (230, 3, 255)  # RGB color for light purple (pinkish)
        thickness = 15  # Thickness of the line

        # Draw the line on the overlay
        # Draw the line on the overlay with sharp edges (no anti-aliasing)
        cv2.line(overlay, (x1_center, y1_bottom), (x2_center, y2_bottom), light_purple, thickness, lineType=cv2.LINE_8)

        # Blend the overlay with the original frame (transparency of 0.4)
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blending the overlay with the frame


# Display the deviation angle and direction
def display_angle_direction(frame, angle, direction):
    if angle is not None and direction is not None:
        #cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Deviation: {angle:.2f} degrees", (15, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (33, 237, 54),
                    2)
        cv2.putText(frame, f"Direction: {direction}", (15, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (33, 237, 54), 2)


# Main function to run the entire process
def main():
    model_path = r"C:\Users\DELL\PycharmProjects\ObjectDetection\models\custom_best(9.10.2024).pt"
    video_path = r"C:\Users\DELL\PycharmProjects\ObjectDetection\videos\Demo Videos\C4076.mp4"
    output_path = r"C:\Users\DELL\PycharmProjects\ObjectDetection\videos\Output Videos\angle_tracked_C4076.mp4"
    bounce_coords_path = r"C:\Users\DELL\PycharmProjects\ObjectDetection\videos\Output Videos\bounce_coords_C4076.txt"

    # Initialize video and model
    model = load_model(model_path)
    desired_width, desired_height = 400, 500
    cap, video_writer, fps = load_video(video_path, desired_width, desired_height, output_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        exit()

    frame = cv2.resize(frame, (desired_width, desired_height))
    mask = select_ground_area(frame, desired_width, desired_height)

    paused = True
    stump_boxes = []
    trajectory_coords = []

    # Video loop
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read.")
                break
            frame = cv2.resize(frame, (desired_width, desired_height))

        frame_copy, angle, direction = process_frame(frame, mask, model, stump_boxes, trajectory_coords,
                                                     bounce_coords_path, cap)
        draw_trajectory(frame_copy, trajectory_coords)
        draw_stump_line(frame_copy, stump_boxes)
        display_angle_direction(frame_copy, angle, direction)

        if paused:
            cv2.putText(frame_copy, "Paused - Select Stump (Press 's')", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        else:
            cv2.putText(frame_copy, "Press 'p' to pause", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display and write frame
        cv2.imshow("Video", frame_copy)
        video_writer.write(frame_copy)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif paused and key == ord('s'):
            box = cv2.selectROI("Select Stumps", frame_copy, fromCenter=False, showCrosshair=True)
            stump_boxes.append(box)
            if len(stump_boxes) == 2:
                paused = False
            cv2.destroyWindow("Select Stumps")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
