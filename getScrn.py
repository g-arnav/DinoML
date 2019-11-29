import os
import cv2

def getScrn():
    os.system("screencapture -R60,125,600,150 holder.png")
    im = cv2.imread('holder.png', 0)
    im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
    im = im / 255.0
    return im
