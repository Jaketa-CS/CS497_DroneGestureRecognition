from djitellopy import Tello

import numpy as np
import cv2 as cv
import mediapipe as mp
from keras import models
import tensorflow as tf
import math

np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)

model = models.load_model("model.h5")
model.load_weights(tf.train.latest_checkpoint("checkpoints"))

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

d = Tello()
d.connect()
d.streamon()
n = 3
frame_read = d.get_frame_read()
cv.namedWindow('Live', cv.WINDOW_NORMAL)


with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while True:
        img = d.get_frame_read().frame
        img = cv.flip(img, 1)
        img = cv.resize(img, (500, 400))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        blank = np.zeros(img.shape)


        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
             #   mp_draw.draw_landmarks(blank, hand, mp_hands.HAND_CONNECTIONS,
              #                         mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
               #                        mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                # extract the x and y coordinates of the landmarks:
                #calculate the bounding box as follows
                x_min, y_min = int(hand.landmark[0].x * img.shape[1]), int(hand.landmark[0].y * img.shape[0])
                x_max, y_max = x_min, y_min

                for landmark in hand.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    if x < x_min:
                        x_min = x-20
                    elif x > x_max:
                        x_max = x+20
                    if y < y_min:
                        y_min = y-30
                    elif y > y_max:
                        y_max = y-10
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                #hand_img = blank[y_min:y_max, x_min:x_max]
               # cv.imshow("Live", hand_img)
                hand_img = cv.resize(hand_img, (100, 200))
                """
                array_img = np.array(hand_img)
                array_img = np.expand_dims(array_img, axis=0)
                network_output = model.predict(array_img)

                temp = [int(x) for x in network_output[0]]
                print("nets: ", network_output)
                cv.setWindowTitle('Live', str(np.argmax(network_output)))
                if math.floor(network_output[0][0]) == 1:
                    print("--------------------------------------------------down\n")
                    print("--------net: ", network_output)
                    # d.move_down(n)

                if math.floor(network_output[0][1]) == 1:
                    print("--------------------------------------------------forward\n")
                    print("--------net: ", network_output)
                # d.move_forward(n)

                if math.floor(network_output[0][2]) == 1:
                    print("--------------------------------------------------left\n")
                    print("--------net: ", network_output)
                    # d.move_left(n)

                if math.floor(network_output[0][3]) == 1:
                    print("--------------------------------------------------right\n")
                    print("--------net: ", network_output)
                    # d.move_right(n)

                if math.floor(network_output[0][4]) == 1:
                    print("--------------------------------------------------land\n")
                    print("--------net: ", network_output)
                # d.land()

                if math.floor(network_output[0][5]) == 1:
                    print("--------------------------------------------------up\n")
                    print("--------net: ", network_output)
                """

        cv.imshow("Live", img)
       # array_img = np.array(img)
        #array_img = np.expand_dims(array_img, axis=0)
        #network_output = model.predict(array_img)

        #temp = [int(x) for x in network_output[0]]
        #print("nets: ", network_output)

       # cv.setWindowTitle('Live', str(np.argmax(network_output)))
        key = cv.waitKey(50)
        if key == ord('p'):
            cv.waitKey(0)
        if key == ord('q'):
            break

cv.destroyAllWindows()
d.streamoff()
d.end()  # ??