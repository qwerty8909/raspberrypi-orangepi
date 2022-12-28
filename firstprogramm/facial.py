import cv2
from gpiozero import Buzzer
from time import sleep

buzzer = Buzzer(17)

face_cascade = cv2.CascadeClassifier("/home/pi/project/opencv-4.1.0/data/haarcascades/cascade.xml")

def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # You can also specify minSize and maxSize

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),
                            color=(0, 255, 0), thickness=5)
        face = frame[y : y+h, x : x+w]
#        print ("Face detect")
        buzzer.on()
        sleep(0.01)
        buzzer.off()
        sleep(0.01)
    return frame

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("No stream :(")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))

# list of FourCC video codes: https://softron.zendesk.com/hc/en-us/articles/207695697-List-of-FourCC-codes-for-video-codecs

#output = cv2.VideoWriter("assets/6_facial_detection.mp4",
#            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
#            fps=fps, frameSize=(width, height))

while(True):
    ret, frame = stream.read()
    if not ret:
        print("No more stream :(")
        break

    frame = detect_features(frame)
#    output.write(frame)
    cv2.imshow("Webcam!", frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows() #!
