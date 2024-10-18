# Custom Cricket Ball Tracking

## Objective:

To track the trajectory of a cricket ball in videos using a custom-trained model.

## Process Flow

1. **Train- Prepare Video Data :** Stitch together 5 cricket videos with the same location and lighting to maintain consistency in data.
1. **Extract Frames**
- **Action**: Use RoboFlow to extract every frame from the stitched video.
- **Purpose**: To ensure the dataset includes individual frames, as this helps in precise annotation.
3. **Annotate Ball in Frames**
- **Action**: Using RoboFlow, manually label the cricket ball in every frame. This helps the model later on to "learn" what the ball looks like across various positions, angles, and lighting conditions.
- **Tools**: RoboFlow’s annotation feature.
4. **Generate Custom Dataset API in RoboFlow**
- **Action**: Once annotation is complete, RoboFlow allows you to export the dataset via an API. 

  The API contains details about dataset and annotations. This is useful because it automates downloading the dataset, instead of manually downloading and uploading files.

- **Purpose**: This API allows integration with other platforms like Google Colab for model training. This API will automatically pull the dataset into your Colab environment, ready for training.
5. **Train Custom Dataset API in Google Colab**
- **Action**: Use Google Colab to train the dataset using a YOLO model. Leverage Colab’s free GPU to speed up the training process. YOLO will train for 100 iterations to learn patterns from our custom data.
- **Outcome**: Once the training process is complete, we'll have a custom Trained YOLO model that is specialized in detecting cricket balls.
6. **Execute Cricket Ball Tracking Code**
- **Action**: Using the custom model in our cricket ball tracking code to monitor the ball’s trajectory across frames.
- **Outcome**: Detect and track the ball’s movement with high accuracy in videos.
7. **Test Videos**
- **Action**: Test the model on videos with similar lighting and locations that were not annotated.
- **Purpose**: Validate the model’s accuracy in detecting and tracking the ball in new, unseen videos.

## Summary of Algorithms Used:

1. **YOLO (You Only Look Once)**:
- Object detection algorithm using a CNN that detects the cricket ball in each frame.
- YOLO divides an image into a grid and assigns bounding boxes to potential objects (in our case, the cricket ball). 
2. **Backpropagation in CNN(YOLO Training)**:
- Core algorithm for training the YOLO model via error correction and weight updates.
- we run training for 100 epochs, meaning the dataset will be fed through the network 100 times to improve learning.
3. **Image Preprocessing**:
- **Image Resizing**: Before feeding images to YOLO, they are resized to a consistent resolution (e.g., 640x640 in your setup). This helps in speeding up the training and makes sure the model receives consistent input sizes.


This project uses a custom-trained YOLOv8 model to detect and track a cricket ball in video footage. It processes each video frame to detect the ball, draws bounding boxes, and tracks the ball's trajectory, saving the processed output video and logging the ball's coordinates.

## Installation

1. **Clone this repository:**

    ```sh
    git clone https://github.com/yourusername/CustomBallTracking.git
    cd CustomBallTracking
    ```

2. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```


## Usage

1. **Run the cricket ball detection script:**

    You can run the ball tracking script directly using:

    ```sh
    python yolo_custom_ball_tracking.py
    ```

2. **Input your video:**

    - Place your input video (e.g., cricket match footage) in the appropriate directory as specified in the script, or modify the `input_video_path` in the code.

3. **View the output:**

    - The processed video with bounding boxes, ball trajectory, and annotations will be saved in the `output_videos_custom_yolo` directory.
    - The ball's center coordinates will be logged in the `ball_coordinates.log` file.
    - Optionally, the trajectory data can be saved as a JSON file.

4. **Customize the Detection Parameters:**

    - If needed, adjust the confidence threshold, video paths, or other parameters directly in the `yolo_custom_ball_tracking.py` script.

5. **View Logs and Results:**

    - Ball center coordinates and trajectory information are logged for each frame in `ball_coordinates.log`.
    - The output video with the cricket ball tracking can be found in the specified output directory.

## How It Works

1. The script uses a custom-trained **YOLOv8** model to detect and track the cricket ball in a video.
2. **OpenCV** is used to process the video frames and draw bounding boxes, center points, and trajectory lines.
3. **Logging** is set up to store the ball's center coordinates for each frame in a log file.
4. The detected ball's trajectory is visually represented by drawing a fading line for older frames to illustrate movement.
5. The processed video, including all visualizations, is saved in the specified output folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request if you have improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ultralytics YOLOv8** for the object detection model.
- **OpenCV** for video processing and visualizations.
- **Python Logging** for tracking and logging ball coordinates.
- **NumPy** for handling numerical computations.
- **torch** and **PyTorch** for model inference.

