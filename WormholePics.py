import WormholeRayTracer as wrmhole
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit
from numba.core import types
from numba.typed import Dict
from numba import njit

# Inladen foto's
print('Reading in pictures...')
img_saturn    = cv2.imread('four400.png')
img_gargantua = cv2.imread('negfour400.png')
# print('here1')
# print(img_gargantua.shape)
# print(len(img_saturn))

#Maak lijsten om dichtste te zoeken
vertical   = len(img_saturn)     #1024
horizontal = len(img_saturn[0])  #2048

theta_list = list()
for teller in range(0, vertical):
    theta = (np.pi/vertical) * teller #- np.pi
    theta_list.append(theta)

phi_list =list()
for teller in range(0, horizontal):
    phi   = (2*np.pi/horizontal) * teller #+ np.pi
    # Nulpunt in het midden van het scherm zetten:
    # if phi > 2*np.pi:
    #     phi = phi - 2*np.pi
    phi_list.append(phi)
# print('here2')

key_ty = types.Tuple((types.float64, types.float64))
# @njit
def photo_to_sphere(photo):
    """
    Give the pixels of the pictures a spherical coordinate
    Input:  - photo: de pixels van de photo in sferische coordinaten
    Output: - dict: een dictionary met als sleutel (theta, phi) en als waarde
              de RGB-value van de bijbehorende pixel
    """

    dict = Dict.empty(key_type = key_ty, value_type = types.float64[:],)
    vertical   = len(photo)     #1024
    horizontal = len(photo[0])  #2048
    for row in range(0, vertical):
        for column in range(0, horizontal):
            theta = (np.pi/vertical) * row #- np.pi
            phi   = (2*np.pi/horizontal) * column #+ np.pi
            # Nulpunt in het midden van het scherm zetten:
            # coordinate = types.Tuple(theta, phi) #Tuple with angles that will be used as key
            pixel      = np.array([photo[row][column]]) #RGB-values
            dict[(theta, phi)] = pixel

    return dict

@jit
def decide_universe(photo, saturn, gargantua):
    """
    Decides whether ray is in Saturn or Gargantua universe and accesses the
    according function to determine the RGB values of the pixels.
    Input:  - photo:     solved ray tracer
            - saturn: spherical picture of the Saturn side
            - gargantua: spherical picture of the other side
    Output: - picture:   Matrix with RGB values for cv2
    """
    picture = []
    for rij in range(len(photo)):
        row = []
        for kolom in range(len(photo[0])):
            if photo[rij][kolom][0] < 0:
                # pixel = ray_to_rgb((photo[rij][kolom][1], photo[rij][kolom][2]), gargantua)
                ph = photo[rij][kolom][1]
                th = photo[rij][kolom][2]

                theta_near = min(theta_list, key=lambda x: distance(x, th))
                phi_near = min(phi_list, key=lambda x: distance(x, ph))
                nearest = (theta_near, phi_near)
                pixel = gargantua[nearest]
            else:
                # pixel = ray_to_rgb((photo[rij][kolom][1], photo[rij][kolom][2]), saturn)
                ph = photo[rij][kolom][1]
                th = photo[rij][kolom][2]

                theta_near = min(theta_list, key=lambda x: distance(x, th))
                phi_near = min(phi_list, key=lambda x: distance(x, ph))
                nearest = (theta_near, phi_near)
                pixel = gargantua[nearest]
            [[R, G, B]] = pixel
            row.append([R, G, B])
        picture.append(np.array(row))
    # img = cv2.cvtColor(np.array(picture, np.float32), 1)
    return np.array(picture)

    # print('here 4')

def distance(x, position):
    """
    Define a distance function for closest neighbour
    """
    dist = abs(x-position)

    return dist


def ray_to_rgb(position, saturn):
    """
    Determines values of the pixels for the rays at the Saturn side.
    Input:  - position: tuple of theta and phi angles: [theta, phi]
            - saturn: spherical picture of the Saturn side
    Output: - List with RBG-values of corresponding pixel of the Saturn picture
    """
    p, t = position

    theta_near = min(theta_list, key=lambda x: distance(x, t))
    phi_near = min(phi_list, key=lambda x: distance(x, p))
    nearest = (theta_near, phi_near)
    RGB = saturn[nearest]

    return RGB


if __name__ == '__main__':
    saturn      = photo_to_sphere(img_saturn)
    print('Saturn image loaded.')
    gargantua   = photo_to_sphere(img_gargantua)
    print('Gargantua image loaded.')

    raytracer = wrmhole.wormhole_with_symmetry(tijd=22, initialcond = [20, np.pi, np.pi/2], Nz=200, Ny=400, Par=[0.43/1.42953, 1, 0])
    # print(raytracer.shape)
    print('Ray tracer solution loaded.')
    print('Starting image placing process...')
    # @jit(nopython=True)
    pic = decide_universe(raytracer, saturn, gargantua)
    print(pic)
    print('Image placing completed.')
    print('Saving picture')
    path = os.getcwd()
    cv2.imwrite(os.path.join(path, 'Pictures_test_x=0.5.png'), pic)
    print('Picture saved')
