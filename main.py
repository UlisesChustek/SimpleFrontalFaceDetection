import pathlib
import cv2

def load_classifier(classifier_path):
    """Load the face detection classifier"""
    classifier = cv2.CascadeClassifier(classifier_path)
    if classifier.empty():
        raise ValueError(f"Error loading classifier: {classifier_path}")
    return classifier

def open_camera(camera_index):
    """Open the default camera"""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise ValueError(f"Error opening camera: {camera_index}")
    return camera

def detect_faces(frame, classifier, scale_factor=1.1, min_neighbors=5, min_size=(30,30), flags=cv2.CASCADE_SCALE_IMAGE):
    """Detect faces in the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size, flags=flags)
    return faces

def draw_rectangles(frame, faces):
    """Draw a rectangle around each detected face"""
    for (x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0), 2)

def display_frame(frame):
    """Display the resulting image"""
    cv2.imshow("Faces", frame)

def save_frame(frame, filepath):
    """Save the frame to a file"""
    cv2.imwrite(filepath, frame)

def main():
    # Load the face detection classifier
    classifier_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = load_classifier(classifier_path)

    # Open the default camera
    camera = open_camera(0)

    # Capture and process frames from the camera
    while camera.isOpened():
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error reading frame from camera")
            break

        # Detect faces in the frame
        faces = detect_faces(frame, classifier)

        # Draw a rectangle around each detected face
        draw_rectangles(frame, faces)

        # Display the resulting image
        display_frame(frame)

        # Check if the user has pressed the "q" key
        if cv2.waitKey(1) == ord("q"):
            # If the "q" key has been pressed, exit the loop
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
