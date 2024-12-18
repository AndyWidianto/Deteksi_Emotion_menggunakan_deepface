import cv2 
import cv2.data
from deepface import DeepFace 

face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

def detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray_frame, scaleFactor=1.1,minNeighbors=5, minSize=(30,30))
    return faces

def drawbox(frame):
    for (x, y, w, h) in detection(frame):
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']
        # gender = result[0]['gender']


        test_x = x
        test_y = y + h - 10

        cv2.rectangle(frame, (x,y),(x+w , y+h), (255, 0, 0 ), 1)
        cv2.putText(frame, f"Emotion : {emotion}", (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 0, 0),1)
        # cv2.putText(frame, gender, (test_x , test_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 0, 0),1)

def CloseWindows():
    camera.release()
    cv2.destroyAllWindows()


while True:
    ret, frame = camera.read()

    drawbox(frame)

    cv2.imshow('Deteksi emosi wajah', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        CloseWindows()