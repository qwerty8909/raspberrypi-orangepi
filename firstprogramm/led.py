from gpiozero import LED
from time import sleep

backleft = LED(26)
backcenter = LED(19)
backright = LED(13)
left = LED(6)
center = LED(5)
right = LED(21)
forleft = LED(20)
forcenter = LED(16)
forright = LED(12)

while True:
    backleft.on()
    sleep(0.1)
    backleft.off()
    sleep(0.1)
    backcenter.on()
    sleep(0.1)
    backcenter.off()
    sleep(0.1)
    backright.on()
    sleep(0.1)
    backright.off()
    sleep(0.1)
    left.on()
    sleep(0.1)
    left.off()
    sleep(0.1)
    center.on()
    sleep(0.1)
    center.off()
    sleep(0.1)
    right.on()
    sleep(0.1)
    right.off()
    sleep(0.1)
    forleft.on()
    sleep(0.1)
    forleft.off()
    sleep(0.1)
    forcenter.on()
    sleep(0.1)
    forcenter.off()
    sleep(0.1)
    forright.on()
    sleep(0.1)
    forright.off()
    sleep(0.1)
