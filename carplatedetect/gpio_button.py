from gpiozero import Button
import os

button = Button(2)
button.wait_for_press()
print('You pushed me')
os.system("sudo shutdown -h now")
