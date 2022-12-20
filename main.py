import pathlib
import cv2

# Load the face detection classifier
try:
    clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except cv2.error as e:
    print(f"Error loading classifier: {e}")
    exit()

# Set up a video capture using the default camera
try:
    camera = cv2.VideoCapture(0)
except cv2.error as e:
    print(f"Error opening camera: {e}")
    exit()

while True:
    # Read a frame from the video capture
    ret, frame = camera.read()
    if not ret:
        print("Error reading frame from camera")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around each detected face
    for (x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y),(x+width, y+height),(255,255,0),2)
        
    # Display the resulting image
    cv2.imshow("Faces",frame)
    
    # Check if the user has pressed the "q" key
    if cv2.waitKey(1) == ord("q"):
        # If the "q" key has been pressed, exit the loop
        break

# Release the video capture and close all windows
camera.release()
cv2.destroyAllWindows()