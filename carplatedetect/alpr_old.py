import cv2
import pytesseract
from time import sleep
from gpiozero import Buzzer, LED
import pandas as pd
from datetime import datetime
import os

timestr = datetime.utcnow().strftime("%S%f")
buzzer = Buzzer(17)
led = LED(21)

base = pd.read_csv('/home/pi/project/base.txt', header=None)
base = base.iloc[:,0]

RTSP_URL = 'rtsp://admin:Password1@192.168.51.249:554/live/main'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

carplate_cascade = cv2.CascadeClassifier('/home/pi/project/haarcascade_russian_plate_number.xml')

def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray, (5,5), 0) 
#    edged = cv2.Canny(blur, 10, 200) 
    carplates = carplate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

    for (x, y, w, h) in carplates:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=5)
        carplate = frame[y : y+h, x : x+w] # carplate = frame[y+15 : y+h-15, x+15 : x+w-15]
        
        cv2.imwrite(timestr+".png",carplate)
        
        carplate_bil = cv2.bilateralFilter(carplate, 15, 75, 75)
        carplate_res = cv2.resize(carplate, None , fx = 2 , fy = 2 ,interpolation = cv2.INTER_CUBIC)
        carplate_med = cv2.medianBlur(carplate, 5)
        carplate_gau = cv2.GaussianBlur(carplate, ( 5 , 5 ), 0 )
        text_bil = pytesseract.image_to_string(carplate_bil, config = '-l rus --psm 13 --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789')
        text_res = pytesseract.image_to_string(carplate_res, config = '-l rus --psm 13 --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789')
        text_med = pytesseract.image_to_string(carplate_med, config = '-l rus --psm 13 --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789')
        text_gau = pytesseract.image_to_string(carplate_gau, config = '-l rus --psm 13 --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789')
        for line in base:
            if line in text_res or line in text_bil or line in text_med or line in text_gau:
#                with open('/home/pi/project/result.txt', 'a') as file:
#                    file.write(text_res + text_bil + text_med + text_gau + '\n')
                buzzer.on()
                led.on()
                sleep(0.01)
                buzzer.off()
                led.off()
                sleep(0.01)
#        print ("Number detect", text_bil, text_res, text_med, text_gau)
    return frame

stream = cv2.VideoCapture(1)
#stream = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) #закомментируй строки с разрешением
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #1280, 640
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) #720, 480, 360
stream.set(cv2.CAP_PROP_FPS, 15) #30, 20, 15, 10
#stream.set(cv2.CV_CAP_PROP_BRIGHTNESS, 100)

while True:
    ret, frame = stream.read()
    frame = detect_features(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()

'''
def carplate_extract(image):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects:
        carplate_img = image[y+15:y+h-10 ,x+15:x+w-20]
    return carplate_img

def text_extract(image):
    list_text = []
    for a in [8, 9, 13]:
        text = pytesseract.image_to_string(image,config = f'-l rus --psm {a} --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789')
        text = ('\n'.join(text.split('\n')[:-1])) # избавимся от лишних переносов строки 
        if len(text) >= 6:
            list_text.append (text)
    return list_text
'''