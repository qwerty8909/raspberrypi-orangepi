import os
from time import sleep

os.system("echo 35 >/sys/class/gpio/export")
os.system("echo 0 >/sys/class/gpio/gpio35/value")
sleep(1.5)
os.system("echo 1 >/sys/class/gpio/gpio35/value")
sleep(1.5)
os.system("echo 0 >/sys/class/gpio/gpio35/value")
sleep(1.5)
os.system("echo 1 >/sys/class/gpio/gpio35/value")