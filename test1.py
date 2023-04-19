import cv2 as cv
import signal
from djitellopy import Tello
import sys

def safetyhandler(sig, frame):
    print('EMERGENCY!!!')
    d.emergency()
    sys.exit(0)


d = Tello()
signal.signal(signal.SIGINT, safetyhandler)


d.connect()

d.takeoff()
d.move_up(150)
d.move_forward(100)
d.move_down(100)
d.move_left(100)
d.move_up(100)
d.move_back(100)
d.move_down(100)
d.rotate_clockwise(90)
d.move_up(100)
d.move_forward(100)
d.move_down(100)
d.move_left(100)
d.move_up(100)
d.move_back(100)
d.move_down(100)
d.land()


#d.streamon()
#frame_read = tello.get_frame_read()
#
#while True:
#    img = tello.get_frame_read().frame
#    img = cv2.resize(img,(360,240))
#    cv2.imshow('{}%'.format(tello.get_battery()),img)
#    val = cv2.waitKey(1)
#    if val & 0xFF ==  ord('q'):
#        break
#
#tello.streamoff()
d.end() # ??
