import cv2 as cv
from djitellopy import Tello
import sys
from keras import models
import tensorflow as tf
import numpy as np
import math
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)

model = models.load_model("model.h5")
model.load_weights(tf.train.latest_checkpoint("checkpoints"))

d = Tello()
d.connect()

d.streamon()
n= 3
frame_read = d.get_frame_read()
#cv.namedWindow('Live', cv.WINDOW_NORMAL)
#d.takeoff()
while True:
    img = d.get_frame_read().frame
    img = cv.resize(img,(500,400))
    cv.imshow("Live",img)
    array_img = np.array(img)
    array_img = np.expand_dims(array_img, axis = 0)
    network_output = model.predict(array_img)
   # print("net1: ", network_output)

    temp = [int(x) for x in network_output[0]]
    #network_output = temp[0]
   # print("len: ", len(network_output))
    print("nets: ", network_output)

    if math.floor(network_output[0][1]) == 0:
        print("--------------------------------------------------down\n")
        print("--------net: ", network_output)
        #d.move_down(n)
        
    if math.floor(network_output[0][1]) == 1:
        print("--------------------------------------------------forward\n")
        print("--------net: ", network_output)
       # d.move_forward(n)

    if math.floor(network_output[0][2]) ==2: 
        print("--------------------------------------------------left\n")
        print("--------net: ", network_output)
        #d.move_left(n)
    
    if math.floor(network_output[0][3]) ==3:
        print("--------------------------------------------------right\n")
        print("--------net: ", network_output)
        #d.move_right(n)
        
    if math.floor(network_output[0][4]) == 4:
        print("--------------------------------------------------land\n")
        print("--------net: ", network_output)
       # d.land()    

    if math.floor(network_output[0][5]) == 5:
        print("--------------------------------------------------up\n")
        print("--------net: ", network_output)
       # d.move_up(n)
       
    val = cv.waitKey(1)
    if val & 0xFF ==  ord('q'):
        break
cv.destroyAllWindows()
d.streamoff()
d.end() # ??