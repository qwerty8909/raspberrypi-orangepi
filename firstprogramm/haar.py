import cv2
from gpiozero import Buzzer
from gpiozero import LED
from time import sleep

buzzer = Buzzer(17)
backleft = LED(26)
backcenter = LED(19)
backright = LED(13)
left = LED(6)
center = LED(5)
right = LED(21)
forleft = LED(20)
forcenter = LED(16)
forright = LED(12)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "dron02.xml")
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # You can also specify minSize and maxSize
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),
                            color=(0, 255, 0), thickness=5)
        face = frame[y : y+h, x : x+w]
        gray_face = gray[y : y+h, x : x+w]
#        print ("Dron detect", x, y)

        if x < 200 and y < 133:
            backleft.on()
            sleep(0.1)
            backleft.off()
            sleep(0.1)
        if x < 200 and y > 133 and y < 266:
            backcenter.on()
            sleep(0.1)
            backcenter.off()
            sleep(0.1)
        if x < 200 and y > 266:
            backright.on()
            sleep(0.1)
            backright.off()
            sleep(0.1)
        if x > 200 and x < 400 and y < 133:
            left.on()
            sleep(0.1)
            left.off()
            sleep(0.1)
        if x > 200 and x < 400 and y > 133 and y < 266:
            center.on()
            sleep(0.1)
            center.off()
            sleep(0.1)
        if x > 200 and x < 400 and y > 266:
            right.on()
            sleep(0.1)
            right.off()
            sleep(0.1)
        if x > 400 and y < 133:
            forleft.on()
            sleep(0.1)
            forleft.off()
            sleep(0.1)
        if x > 400 and y > 133 and y < 266:
            forcenter.on()
            sleep(0.1)
            forcenter.off()
            sleep(0.1)
        if x > 400 and y > 266:
            forright.on()
            sleep(0.1)
            forright.off()
            sleep(0.1)

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

while(True):
    ret, frame = stream.read()
    if not ret:
        print("No more stream :(")
        break
    
    frame = detect_features(frame)
#    cv2.imshow("Webcam!", frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows() #!
