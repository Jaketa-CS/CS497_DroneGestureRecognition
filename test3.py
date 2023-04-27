import cv2 as cv
import signal
from djitellopy import Tello
import sys
import threading
import numpy as np
from time import sleep

running = True

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict(winSize = (25, 25),
                maxLevel = 7,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def safetyhandler(sig, frame):
    print('EMERGENCY!!!')
    d.emergency()
    sys.exit(0)

def videothread():
    global running
    #sleep(20) # for when no movement is wanted
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
    #d.flip_forward() # yeet
    d.move_left(100)
    d.move_up(100)
    #d.flip_back() # yeet
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
# Inspired by our hw7 + https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

color = np.random.randint(0, 255, (100, 3))
oldpoints = []
oldframe = d.get_frame_read().frame
oldpoints = cv.goodFeaturesToTrack(cv.cvtColor(oldframe, cv.COLOR_BGR2GRAY), mask=None, **feature_params)
mask = np.zeros_like(oldframe)
while running:
    frame = d.get_frame_read().frame
    #frame = cv.resize(frame,(360,240))

    if len(oldpoints) > 0:
        newpoints, st, err = cv.calcOpticalFlowPyrLK(
                cv.cvtColor(oldframe, cv.COLOR_BGR2GRAY),
                cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
                np.array(oldpoints, dtype=np.float32),
                None,
                **lk_params
                )
        if newpoints is not None:
            goodnew = newpoints[st==1]
            goodold = oldpoints[st==1]

        for i, (new, old) in enumerate(zip(goodnew, goodold)):
            a, b = new.ravel()
            c, e = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(e)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

        cv.imshow('LiveFeed',img)
        cv.setWindowTitle('LiveFeed', '{}%'.format(d.get_battery()))
        val = cv.waitKey(1)
        if val & 0xFF ==  ord('q'):
            break

        oldframe = frame.copy()
        oldpoints = goodnew.reshape(-1, 1, 2)

cv.destroyAllWindows()


# join thread
vthread.join()

d.streamoff()
d.end() # ??
