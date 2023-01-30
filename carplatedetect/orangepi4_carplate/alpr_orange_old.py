import os
import cv2
import pytesseract
import pandas as pd
from gosnomer import normalize
from time import sleep
from datetime import datetime

# пин 13
os.system("echo 35 >/sys/class/gpio/export")
os.system("echo in >/sys/class/gpio/gpio35/direction")
# путь к базе номеров
base = pd.read_csv('/home/orangepi/project/base.txt', header=None)
base = base.iloc[:, 0]
# путь к классификатору
carplate_cascade = cv2.CascadeClassifier('/home/orangepi/project/haarcascade_russian_plate_number.xml')


# функция индикации
def gate_open():
    os.system("echo out >/sys/class/gpio/gpio35/direction")
    sleep(1.5)
    os.system("echo in >/sys/class/gpio/gpio35/direction")


# функция распознавания
def detect_features(frame):
    # переводим в серый
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # для более быстрого распознавания рамки номера уменьшаем картинку
    gray_res = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_LINEAR)

    # применяем каскад
    carplates = carplate_cascade.detectMultiScale(gray_res, scaleFactor=1.05, minNeighbors=5)
    for (x, y, w, h) in carplates:
        # т.к. картинку сжимали в 2 раза, то координаты на не сжатой картинке будут больше в 2 раза
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        # список для распознанных номеров
        list_text = []
        # получили первое изображение номера
        first_carplate = gray[y + 15:y + h - 10, x + 15:x + w - 20]
        cv2.imwrite('first_carplate'+str(cap)+'.png', first_carplate)

        # трансформируем номер в прямоугольник
        # Define the 4 points for the perspective transformation
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([[0, 0], [w, -10], [w, h + 20], [0, h]])
        # Get the perspective transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        # Apply the perspective transformation to the license plate
        carplate = cv2.warpPerspective(first_carplate, transformation_matrix, (w, h))

        # номер авто в png
        cv2.imwrite('carplate'+str(cap)+'.png', carplate)
        # номер увеличиваем в 2 раза, применяем несколько фильтров, получаем второе изображение номера
        resize_plate = cv2.resize(carplate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gauss = cv2.GaussianBlur(resize_plate, (5, 5), 0)
        blur = cv2.medianBlur(gauss, 5)
        carplate_res = cv2.bilateralFilter(blur, 11, 15, 15)
        # перебираем изображения номера
        for plate in [carplate, carplate_res]:
            # перебираем несколько 'psm' при распознавания
            for psm in [6, 13]:
                # распознаем
                text = pytesseract.image_to_string(plate,
                                                   config=f'-l rus --psm {psm} --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789')
                # удаляем лишние символы
                text = ('\n'.join(text.split('\n')[:-1]))
                # если длина номера больше 6 символов
                if len(text) >= 6:
                    try:
                        # нормализуем и в список
                        norm_text = normalize(text)
                        list_text.append(norm_text)
                    except ValueError:
                        # если не нормализовали, то в список
                        list_text.append(text)
        # пишем лог со всеми распознанными номерами
        timestr = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
        with open('/home/orangepi/project/result.txt', 'a') as file:
            file.write(timestr + ' - ' + str(cap) + ' - ' + ' '.join(list_text) + '\n')
        # ищем соответствие номера в базе с распознанным списком номеров
        for i in base:
            for j in list_text:
                if i in j:
                    with open('/home/orangepi/project/result.txt', 'a') as file:
                        file.write('OK' + '\n')
                    # если да, то оправляем в функцию индикации
                    gate_open()
                    break
    return frame


# выбираем камеру 0 или 1 (-1 любая доступная)
stream = cv2.VideoCapture(0)
# формирум поток кадров 640x360 и 30 кадров в секунду
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
stream.set(cv2.CAP_PROP_FPS, 30)
# stream.set(cv2.CV_CAP_PROP_BRIGHTNESS, 100)

# выбирам необходимый (10) для вывода (30/10 = 3 кадра в секунду)
cap = 1
frameRate = 10
while True:
    ret, frame = stream.read()
    if ret:
        if (cap % frameRate == 0):
            # выводим в консоль лог с выданным кадром
            timestr = datetime.utcnow().strftime("%S")
            print(str(cap) + ' - ' + timestr)
            # сохраняем кадры в png
            cv2.imwrite('frame'+str(cap)+'.png', frame)
            # отправляем в функцию распознавания
            frame = detect_features(frame)
        cap += 1

# закрываем все окна
stream.release()
cv2.destroyAllWindows()

'''
Вместо того, чтобы использовать цикл for для базовой переменной и еще один цикл for для переменной list_text, 
вы можете использовать функцию any(), чтобы проверить, находится ли какой-либо элемент базы в list_text.

Поскольку вы используете одни и те же шаги обработки изображения несколько раз, 
вы можете извлечь их как функцию и вызвать ее вместо многократного повторения одного и того же кода.
'''
