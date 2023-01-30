import cv2
import numpy as np

# Load image
img = cv2.imread('7.jpg')
carplate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

carplates = carplate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
for (x, y, w, h) in carplates:
    carplate = gray[y:y+h, x:x+w]
    # carplate = gray[y+15:y+h-10, x+15:x+w-20]

# Define the 4 points for the perspective transformation
    src_points = np.float32([[0,0], [w,0], [w,h], [0,h]])
    dst_points = np.float32([[20,-30], [w,20], [w,h-20], [20,h+10]])

# Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to the license plate
    warped_plate = cv2.warpPerspective(carplate, M, (w,h))

    src_points_a = np.float32([[0,0], [w,0], [w,h], [0,h]])
    dst_points_a = np.float32([[0,0], [w,-10], [w,h+20], [0,h]])
    P = cv2.getPerspectiveTransform(src_points_a, dst_points_a)
    rep_plate = cv2.warpPerspective(warped_plate, P, (w,h))

# Show the original and transformed images
    cv2.imshow("Transformed", warped_plate)
    cv2.imshow("rep_plate", rep_plate)
cv2.waitKey(0)
