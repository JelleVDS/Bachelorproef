import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import scipy.integrate as integr
import math

def make4color(height=1024, width=2048):
    pic = np.empty([height, width, 3])
    for row in range(height):
        for column in range(width):
            loch = row + height/2
            locp = column
            if row <= height/2 and row%(height/64) > 0:
                if column < width/2 and column%(height/64)>0:
                    pic[row][column] = np.array([255, 0, 0])
                if column >= width/2 and column%(height/64)>0:
                    pic[row][column] = np.array([123, 123, 0])
            if row > height/2 and row%(height/64) > 0:
                locp = column + width/2
                if column < width/2 and column%(height/64)>0:
                    pic[row][column] = np.array([0, 255, 0])
                if column >= width/2 and column%(height/64)>0:
                    pic[row][column] = np.array([0, 0, 255])
    return np.array(pic)

pict = make4color()
print(pict.shape)
img = cv2.cvtColor(np.array(pict, np.float32), 1)
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'fourfull.png'), img)