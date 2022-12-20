# Face Detection Program

This program uses the `cv2` (OpenCV) module to detect faces in a video stream from the default camera.

## Prerequisites

- Python 3
- `cv2` (OpenCV) module
- `pathlib` module

## Usage

To run the program, open a terminal or command prompt, navigate to the directory where the script is located, and enter the following command:

    python face_detection.py


The program will start the video capture and display a window showing the video stream. When it detects a face in the video, it will draw a rectangle around the face.

To stop the program, press the "q" key on your keyboard.

## Notes

- This program uses the default face detection classifier provided by OpenCV. You can use a different classifier by specifying the path to the classifier XML file in the `cv2.CascadeClassifier` constructor.
