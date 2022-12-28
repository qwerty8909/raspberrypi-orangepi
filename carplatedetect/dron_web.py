import cv2
from gpiozero import Buzzer
from gpiozero import LED
from time import sleep

buzzer = Buzzer(17)

forright = LED(26)
forcenter = LED(19)
forleft = LED(13)
right = LED(6)
center = LED(21)
left = LED(20)
backright = LED(16)
backcenter = LED(12)
backleft = LED(5)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "dron02.xml")

def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # You can also specify minSize and maxSize
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),
                            color=(0, 255, 0), thickness=5)
        face = frame[y : y+h, x : x+w]
#        print ("Dron detect", x+(w/2), y+(h/2))

        if x+(w/2) < 213 and y+(h/2) < 213:
            forleft.on()
            sleep(0.1)
            forleft.off()
            sleep(0.1)
        if x+(w/2) < 213 and y+(h/2) >= 213 and y+(h/2) <= 427:
            forcenter.on()
            sleep(0.1)
            forcenter.off()
            sleep(0.1)
        if x+(w/2) < 213 and y+(h/2) > 427:
            forright.on()
            sleep(0.1)
            forright.off()
            sleep(0.1)
        if x+(w/2) >= 213 and x+(w/2) <= 427 and y+(h/2) < 213:
            left.on()
            sleep(0.1)
            left.off()
            sleep(0.1)
        if x+(w/2) >= 213 and x+(w/2) <= 427 and y+(h/2) >= 213 and y+(h/2) <= 427:
            center.on()
            sleep(0.1)
            center.off()
            sleep(0.1)
        if x+(w/2) >= 213 and x+(w/2) <= 427 and y+(h/2) > 427:
            right.on()
            sleep(0.1)
            right.off()
            sleep(0.1)
        if x+(w/2) > 427 and y+(h/2) < 213:
            backleft.on()
            sleep(0.1)
            backleft.off()
            sleep(0.1)
        if x+(w/2) > 427 and y+(h/2) >= 213 and y+(h/2) <= 427:
            backcenter.on()
            sleep(0.1)
            backcenter.off()
            sleep(0.1)
        if x+(w/2) > 427 and y+(h/2) > 427:
            backright.on()
            sleep(0.1)
            backright.off()
            sleep(0.1)

        buzzer.on()
        sleep(0.01)
        buzzer.off()
        sleep(0.01)

    return frame

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
stream.set(cv2.CAP_PROP_FPS, 15)

if not stream.isOpened():
    print("No stream :(")
    exit()

while True:
    ret, frame = stream.read()
    frame = detect_features(frame)
#    cv2.imshow("Webcam!", frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows() #!
