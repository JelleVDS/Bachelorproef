# import WormholeRayTracer as wrmhl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Inladen foto's
print('Reading in pictures...')
img_saturn    = cv2.imread('Saturn.jpg')
img_gargantua = cv2.imread('wormhole.jpg')
# print(img_gargantua.shape)
# print(len(img_saturn))

#Maak lijsten om dichtste te zoeken
vertical   = len(img_saturn)     #1024
horizontal = len(img_saturn[0])  #2048

theta_list = list()
for teller in range(0, 1024):
    theta = (np.pi/vertical) * teller - np.pi
    theta_list.append(theta)

phi_list =list()
for teller in range(0, 2048):
    phi   = (2*np.pi/horizontal) * teller + np.pi
    # Nulpunt in het midden van het scherm zetten:
    if phi > 2*np.pi:
        phi = phi - 2*np.pi
    phi_list.append(phi)

def photo_to_sphere(photo):
    """
    Give the pixels of the pictures a spherical coordinate
        -herschalen
        -reduceren naar Ny x Nz
    Input:  - photo: de pixels van de photo in sferische coordinaten
    Output: - dict: een dictionary met als sleutel (theta, phi) en als waarde
              de RGB-value van de bijbehorende pixel
    """

    dict = {}
    vertical   = len(photo)     #1024
    horizontal = len(photo[0])  #2048
    for row in range(0, vertical):
        for column in range(0, horizontal):
            theta = (np.pi/vertical) * row - np.pi
            phi   = (2*np.pi/horizontal) * column + np.pi
            # Nulpunt in het midden van het scherm zetten:
            if phi > 2*np.pi:
                phi = phi - 2*np.pi

            coordinate = (theta, phi) #Tuple with angles that will be used as key
            pixel      = np.array([photo[row][column]]) #RGB-values
            dict[coordinate] = pixel

    return dict

# ph = photo_to_sphere(img_saturn)
# im = np.array([])
# for element in ph:
#     print(element)
#     theta, phi = element
#     im[theta][phi] = ph[element]
#
# path = os.getcwd()
# cv2.imwrite(os.path.join(path, 'test.png'), im)


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
    for rij in range(len(photo[-1][1][0])):
        row = []
        for kolom in range(len(photo[-1][1][0][0])):
            if photo[-1][1][0][rij,kolom] <= 0:
                pixel = ray_to_gar((photo[-1][1][1][rij][kolom], photo[-1][1][2][rij][kolom]), gargantua)
            else:
                pixel = ray_to_saturn((photo[-1][1][1][rij][kolom], photo[-1][1][2][rij][kolom]), saturn)

            [[R, G, B]] = pixel
            row.append([R, G, B])

        picture.append(np.array(row))
    # img = cv2.cvtColor(np.array(picture, np.float32), 1)
    return np.array(picture)

def distance(x, position):
    """
    Define a distance function for closest neighbour
    """
    dist = abs(x-position)

    return dist


def ray_to_saturn(position, saturn):
    """
    Determines values of the pixels for the rays at the Saturn side.
    Input:  - position: tuple of theta and phi angles: [theta, phi]
            - saturn: spherical picture of the Saturn side
    Output: - List with RBG-values of corresponding pixel of the Saturn picture
    """
    t, p = position
    screen = saturn.keys()
    theta_near = min(theta_list, key=lambda x: distance(x, t))
    phi_near = min(phi_list, key=lambda x: distance(x, p))
    nearest = (theta_near, phi_near)
    # print(nearest)
    RGB = saturn[nearest]
    # print(RGB)

    return RGB

def ray_to_gar(position, gargantua):
    """
    Determines values of the pixels for the rays at the Gargantua side.
    Input:  - position: tuple of theta and phi angles: [theta, phi]
            - gargantua: spherical picture of the other side
    Output: - List with RBG-values of corresponding pixel of the Saturn picture
    """
    t, p = position
    screen = gargantua.keys()
    theta_near = min(theta_list, key=lambda x: distance(x, t))
    phi_near = min(phi_list, key=lambda x: distance(x, p))
    nearest = (theta_near, phi_near)
    RGB = gargantua[nearest]
    # print(RGB)

    return RGB


saturn      = photo_to_sphere(img_saturn)
# np.savez('sat', saturn)
print('Saturn image loaded.')
gargantua   = photo_to_sphere(img_gargantua)
# np.savez('gar', gargantua)
print('Gargantua image loaded.')

raytracer = np.load('ray_solved.npy')
print('Ray tracer solution loaded.')
# print(raytracer)
# saturn = np.load('sat.npz')
# gargantua = np.load('gar.npz')
print('Starting image placing process...')
pic = decide_universe(raytracer, saturn, gargantua)
# print(pic)
# print(pic.shape)
print('Image placing completed.')
# np.savez('picInterstellar', pic)
# print(pic)
print('Saving picture')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'InterstellarWormhole_14_5.png'), pic)
print('Picture saved')
