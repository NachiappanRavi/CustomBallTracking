import os
import subprocess

def extract_frames(input_video, output_directory, fps=60):
    # Define the name of the frames folder
    frames_folder = os.path.join(output_directory, "frames")

    # Create the frames directory if it doesn't exist
    os.makedirs(frames_folder, exist_ok=True)

    # Construct the FFmpeg command
    command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f'fps={fps}',
        os.path.join(frames_folder, 'output_frame_%04d.png')
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Frames extracted to {frames_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    input_video_path = r"C:\Users\Admin\Desktop\ANNO\red\IMG_1176.MOV"  # Replace with your input video path
    output_directory_path = r"C:\Users\Admin\Desktop\ANNO\red"  # Replace with your output directory path
    extract_frames(input_video_path, output_directory_path, fps=60)
