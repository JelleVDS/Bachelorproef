import WormholeRayTracer as wrmhl
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Inladen foto's
img_saturn    = cv2.imread('Saturn.jpg')
img_gargantua = cv2.imread('wormhole.jpg')

# print(len(img_saturn))

def photo_to_sphere(photo):
    """
    Give the pixels of the pictures a spherical coordinate
        -herschalen
        -reduceren naar Ny x Nz

    Input:  - photo: de pixels van de photo in sferische coordinaten
    Output: - matrix: een 2D matrix van dezelfde grootte als photo waarin elk
              element op de eerste plaats [theta, phi] draagt en op de tweede
              plaats de RGB-waarde van de bijbehorende pixel.
    """

    matrix = []
    vertical   = len(photo)     #1024
    horizontal = len(photo[0])  #2048
    for row in range(0, vertical):
        rij = []
        for column in range(0, horizontal):
            theta = (np.pi/vertical) * row - np.pi
            phi   = (2*np.pi/horizontal) * column + np.pi
            # Nulpunt in het midden van het scherm zetten:
            if phi > 2*np.pi:
                phi = phi - 2*np.pi

            # afronding:
            theta = np.round(theta, 3)
            phi   = np.round(phi, 3)

            coordinate = np.array([theta, phi])
            pixel      = np.array([photo[row][column]])
            rij.append(np.array([coordinate, pixel], dtype = object))
        matrix.append(np.array(rij))

    matrix = np.array(matrix)
    return matrix


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
    for rij in range(len(photo[0])):
        row = []
        for kolom in range(len(photo[0][0])):
            if q[0][rij,kolom] <= 0:
                pixel = ray_to_gar([photo[1][rij,kolom], photo[2][rij,kolom]], gargantua)
            else:
                pixel = ray_to_saturn([photo[1][rij,kolom], photo[2][rij,kolom]], saturn)

            row.append(pixel)

        picture.append(row)

    return picture

def ray_to_saturn(position, saturn):
    """
    Determines values of the pixels for the rays at the Saturn side.

    Input:  - position: list of theta and phi angles: [theta, phi]
            - saturn: spherical picture of the Saturn side
    Output: - List with RBG-values of corresponding pixel of the Saturn picture
    """
    # Uitlezen waarden
    theta, phi = position
    # Afronding:
    theta = np.round(theta, 3)
    phi   = np.round(phi, 3)

    for pixel in saturn:
        if [theta, phi] == pixel[0]:
            RGB = pixel[1]
        else:
            raise('No correspondig pixel found')

    return RGB

def ray_to_gar(position, gargantua):
    """
    Determines values of the pixels for the rays at the Gargantua side.

    Input:  - position: list of theta and phi angles: [theta, phi]
            - gargantua: spherical picture of the other side
    Output: - List with RBG-values of corresponding pixel of the Saturn picture
    """
    # Uitlezen waarden
    theta, phi = position
    # Afronding:
    theta = np.round(theta, 3)
    phi   = np.round(phi, 3)

    for pixel in gargantua:
        if [theta, phi] == pixel[0]:
            RGB = pixel[1]
        else:
            raise('No correspondig pixel found')

    return RGB


raytracer   = np.load('ray_solved.npy')
saturn      = photo_to_sphere(img_saturn)
np.save('sat', saturn)
gargantua   = photo_to_sphere(img_gargantua)
np.save('gar', gargantua)
pic         = decide_universe(raytracer, saturn, gargantua)



path = os.getcwd()
cv2.imwrite(os.path.join(path, 'InterstellarWormhole.png'), pic)
