import os
import time
import random

i = 0
s = time.time()
for n in range(200, 250):
    i = 1
    os.system("screencapture -R60,125,600,150 LossPics/{}.png".format(n))
    print n