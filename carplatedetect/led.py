from gpiozero import LED
from time import sleep

forright = LED(26)
forcenter = LED(19)
forleft = LED(13)
right = LED(6)
center = LED(21)
left = LED(20)
backright = LED(16)
backcenter = LED(12)
backleft = LED(5)

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
