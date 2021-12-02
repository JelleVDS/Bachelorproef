import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import scipy.integrate as integr
import math

def diagonal_carth(horizonal, vertical, L):
    """
    Gives the carthesian coordinates of the diagonal line given a dimension in pixels
    Input:  -horizontal: horizontal amount of pixels
            -vertical: vertical amount of pixels
            -L: largest length of the screen
    Output: -vec = vector with the coordinates of the diagonal line in
             carthesian coordinates as tuples
    """
    N = int(math.ceil(np.sqrt(horizontal**2 + vertical**2)))
def make4color(height=400, width=400):
    pic = []
    for row in range(height):
        rij = []
        for column in range(width):
            if row <= height/2 and row%(height/10) != 0:
                if column < width/2 and column%(height/10)!=0 :
                    rij.append(np.array([255, 0, 0]))
                if column >= width/2 and column%(height/10)!=0:
                    rij.append(np.array([123, 123, 0]))
            if row > height/2 and row%(height/10) != 0:
                if column < width/2 and column%(height/10)!=0:
                    rij.append(np.array([0, 255, 0]))
                if column >= width/2 and column%(height/10)!=0:
                    rij.append(np.array([0, 0, 255]))
            if column%(height/10)==0:
                rij.append([0,0,0])
        if row%(height/10) == 0:
            u = [0,0,0]
            rij = [u]*width
        pic.append(np.array(rij))
    return np.array(pic)

pict = make4color()
print(pict.shape)
img = cv2.cvtColor(np.array(pict, np.float32), 1)
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'four400.png'), img)