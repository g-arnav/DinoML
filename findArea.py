import cv2
import os

while(1):
    os.system("screencapture -R60,125,600,150 holder.png")
    im = cv2.imread('holder.png', 0)
    cv2.imshow('Press the esc key to exit', im)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
