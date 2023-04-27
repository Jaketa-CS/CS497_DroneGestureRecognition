# live feed from drone, no video correction/changes
import cv2 as cv
import signal
from djitellopy import Tello
import sys
import threading

running = True

lk_params = dict(winSize = (25, 25),
                maxLevel = 7,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def safetyhandler(sig, frame):
    print('EMERGENCY!!!')
    d.emergency()
    sys.exit(0)

def videothread():
    global running
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
    d.flip_forward() # yeet
    d.move_left(100)
    d.move_up(100)
    d.flip_back() # yeet
    d.move_back(100)
    d.move_down(100)
    d.land()
    running = False

d = Tello()
signal.signal(signal.SIGINT, safetyhandler)

d.connect()
d.streamon()
# start thread
vthread = threading.Thread(target=videothread)
vthread.start()

cv.namedWindow('LiveFeed', cv.WINDOW_NORMAL)

oldframe = d.get_frame_read().frame
while running:
    img = d.get_frame_read().frame
    #img = cv.resize(img,(360,240))

    cv.imshow('LiveFeed',img)
    cv.setWindowTitle('LiveFeed', '{}%'.format(d.get_battery()))
    val = cv.waitKey(1)
    if val & 0xFF ==  ord('q'):
        break
cv.destroyAllWindows()


# join thread
vthread.join()

d.streamoff()
d.end() # ??
