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
def make4color(height=1024, width=2048):
    pic = np.empty([height, width, 3])
    for row in range(height):
        for column in range(width):
            loch = row + height/2
            locp = column
            if row <= height/2 :
                if column < width/2:
                    pic[row][column] = np.array([0, 255, 255])
                if column >= width/2:
                    pic[row][column] = np.array([123, 123, 255])
            if row > height/2:
                locp = column + width/2
                if column < width/2:
                    pic[row][column] = np.array([255, 0, 255])
                if column >= width/2:
                    pic[row][column] = np.array([255, 255, 0])
            theta = loch * np.pi / height
            phi   = locp * 2 * np.pi / width
            while phi>2*np.pi:
                phi = phi - 2*np.pi
            while theta > np.pi:
                theta = theta - np.pi

            x = np.cos(theta) * np.cos(phi)
            y = np.cos(theta) * np.sin(phi)
            z = np.sin(theta)
            if x%0.2 < 0.005 or y%0.2 < 0.005 or z%0.2 < 0.005:
                pic[row][column] = np.array([0, 0, 0])
    return np.array(pic)

pict = make4color()
print(pict.shape)
img = cv2.cvtColor(np.array(pict, np.float32), 1)
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'testNegColorGrid.png'), img)