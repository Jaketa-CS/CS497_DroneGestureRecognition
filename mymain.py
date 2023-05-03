import cv2 as cv
import mediapipe as mp
import time
import uuid
import random
import os

#call hand object detector
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("hello")
cap = cv.VideoCapture(0)
#make directory to save images
#os.mkdir("peace")
#
#detects the hand using the mp_hands.Hands function from the Mediapipe library.
#The max_num_hands parameter is set to 1, indicating that it should detect only one hand,
#the min_detection_confidence parameter is set to 0.8, which means that it should be
#confident that it has detected a hand before it starts tracking it. The min_tracking_confidence
#parameter is set to 0.5, which means that it should be at least 50% confident that it is still tracking the hand.
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        img = cv.flip(frame, 1)
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        #To draw a bounding box around the hand,
        #First extract the coordinates of the hand landmarks.

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                # extract the x and y coordinates of the landmarks:
                #calculate the bounding box as follows
                x_min, y_min = int(hand.landmark[0].x * img.shape[1]), int(hand.landmark[0].y * img.shape[0])
                x_max, y_max = x_min, y_min
                #Inside the for loop that iterates over the detected hands
                # add the following code to extract the bounding box coordinates:
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
                #draw a rectangle around the hand:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #save images
        #cv.imwrite(os.path.join("peace","{}.jpg".format(uuid.uuid1())),img)

        cv.imshow("Frame", img)
        key = cv.waitKey(50)
        if key == ord('p'):
            cv.waitKey(0)
        if key == ord('q'):
            break

cv.destroyAllWindows()
cap.release()

