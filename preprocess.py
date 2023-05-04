import cv2 as cv
import numpy as np
import os
from glob import glob

"""
This script is built to be a one-to-many image preprocessing script
For each single image, many training images can be generated
Currently only brightness adaptations and a Canny edge image is output

File structure assumptions:
Given a source folder,
    which contains a list of folders of topics/tasks/etc,
        each image in each folder will be processed into many

For example:
SRCDIR
  |- Label1
  |    |- img1.jpg
  |    |- img2.jpg
  |    |- img3.jpg
  |- Label2
       |- img3.jpg
       |- img2.jpg
       |- img1.jpg
"""

# environment variables
outdir = 'outimgs'
srcdir = 'inimgs'
brightnessval = 25
brightnessscale = 4

# convert to a valid string of a output image
# this is necessary as this script needs to be run from where the src and
#   out directories live
def outputFilename(img, purpose=''):
    global outdir
    if not purpose:
        return outdir + '/' + d + '/' + os.path.basename(img)
    else:
        return outdir + '/' + d + '/' + purpose + '_' + os.path.basename(img)

# from https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
def change_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

# works fine if the output dir is removed
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# iterate through src directories
for d in os.listdir(srcdir):
    if not os.path.isdir(outdir + '/' + d):
        os.mkdir(outdir + '/' + d)
    if os.path.isdir(srcdir + '/' + d):
        imgpath = srcdir + '/' + d + '/'
        for e in glob(imgpath + '*.jpg'):
            print('looking at {}'.format(e))
            cimg = cv.imread(e)
            # process image
            outimg = cv.Canny(cimg, 50, 150)
            cv.imwrite(outputFilename(e, 'canny'), outimg)
            # brightness versions
            for c in range(-brightnessval, brightnessval):
                outstr = 'dark' + str(c)
                cv.imwrite(
                        outputFilename(e, outstr),
                        change_brightness(cimg, c * brightnessscale))

            # watershed
            #ret, markers = cv.connectedComponents(cv.cvtColor(cimg, cv.COLOR_BGR2GRAY))
            #wimg = cv.watershed(cimg, markers)
            #cv.imwrite(outputFilename(e, 'watershed'), wimg)

