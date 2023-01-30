import cv2
import numpy as np

# Load the image
img = cv2.imread("9.jpg")
carplate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Pre-processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

# Identify the car number
# contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
plate = carplate_cascade.detectMultiScale(thresh, scaleFactor=1.05, minNeighbors=5)
for (x, y, w, h) in plate:
    # car_number_image = gray[y:y+h, x:x+w]
    car_number = gray[y+15:y+h-10, x+15:x+w-20]

    rows, cols = car_number.shape[:2]
    pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])
    pts2 = np.float32([[10,10],[cols-20,10],[10,rows-20]])
    M = cv2.getAffineTransform(pts1,pts2)
    car_number = cv2.warpAffine(car_number,M,(cols,rows))

# Show the image
cv2.imshow('Number Plate', car_number)
cv2.waitKey(0)
cv2.destroyAllWindows()