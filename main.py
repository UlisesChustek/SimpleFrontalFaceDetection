import pathlib
import cv2

clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)#If you want to put more cameras you need to change this value, by default it's 0 means that works with 1 camera#

while True:
    _, frame= camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y),(x+width, y+height),(255,255,0),2)
        
    cv2.imshow("Faces",frame)
    if cv2.waitKey(1) == ord("q"):#If you want to change the key to stop the program you need to change this value#
        break
camera.release()
cv2.destroyAllWindows()