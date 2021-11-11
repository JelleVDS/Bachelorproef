import WormholeRayTracer as wrmhole
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

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
def photo_to_sphere(photo):
    """
    Give the pixels of the pictures a spherical coordinate
    Input:  - photo: de pixels van de photo in sferische coordinaten
    Output: - dict: een dictionary met als sleutel (theta, phi) en als waarde
              de RGB-value van de bijbehorende pixel
    """

    dict = {}
    vertical   = len(photo)     #1024
    horizontal = len(photo[0])  #2048
    for row in range(0, vertical):
        for column in range(0, horizontal):
            theta = (np.pi/vertical) * row #- np.pi
            phi   = (2*np.pi/horizontal) * column #+ np.pi
            # Nulpunt in het midden van het scherm zetten:
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
    for rij in range(len(photo)):
        row = []
        for kolom in range(len(photo[0])):
            if photo[rij][kolom][0] < 0:
                pixel = ray_to_rgb((photo[rij][kolom][1], photo[rij][kolom][2]), gargantua)
            else:
                pixel = ray_to_rgb((photo[rij][kolom][1], photo[rij][kolom][2]), saturn)

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

<<<<<<< HEAD

saturn      = photo_to_sphere(img_saturn)
print('Saturn image loaded.')
gargantua   = photo_to_sphere(img_gargantua)
print('Gargantua image loaded.')
# import numpy as np

raytracer = wrmhole.wormhole_with_symmetry(steps=3000, initialcond = [70, np.pi, np.pi/2], Nz=200, Ny=400, Par=[0.43/1.42953, 8.6, 43])
# print(raytracer.shape)
print('Ray tracer solution loaded.')
print('Starting image placing process...')
pic = decide_universe(raytracer, saturn, gargantua)
# print(pic)
print('Image placing completed.')
print('Saving picture')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'param_70_0.43_8.6_43.png'), pic)
print('Picture saved')

# l = [53.79, 58.05, 96.75]
# W = [0.43, 0.12, 0.55]
# a = [0.043, 4.3, 43]

# for i in range(0, len(l)):
#     print('started with l:'+str(i))
#     raytracer_l = wrmhole.wormhole_with_symmetry(steps=3000, initialcond = [l[i], np.pi, np.pi/2], Nz=200, Ny=400, Par=[0.43/1.42953, 8.6, 4.3])
#
#     pic_l = decide_universe(raytracer_l, saturn, gargantua)
#     path = os.getcwd()
#     cv2.imwrite(os.path.join(path, 'param_'+str(l[i])+'_0.43_8.6_4.3.png'), pic_l)
#     print('pictures saved:'+str(i))

# for i in range(0, len(W)):
#     print('started with W:'+str(i))
#     raytracer_W = wrmhole.wormhole_with_symmetry(steps=3000, initialcond = [70, np.pi, np.pi/2], Nz=200, Ny=400, Par=[W[i]/1.42953, 8.6, 4.3])
#     # raytracer_a = wrmhole.wormhole_with_symmetry(steps=3000, initialcond = [300, np.pi, np.pi/2], Nz=200, Ny=400, Par=[0.43/1.42953, 8.6, a[i]])
#
#     pic_W = decide_universe(raytracer_W, saturn, gargantua)
#     path = os.getcwd()
#     cv2.imwrite(os.path.join(path, 'param_70_'+str(W[i])+'_8.6_4.3.png'), pic_W)
#     print('pictures saved:'+str(i))
#
# for i in range(0, len(a)):
#     print('started with a:'+str(i))
#     raytracer_a = wrmhole.wormhole_with_symmetry(steps=3000, initialcond = [70, np.pi, np.pi/2], Nz=200, Ny=400, Par=[0.43/1.42953, 8.6, a[i]])
#
#     pic_a = decide_universe(raytracer_a, saturn, gargantua)
#     path = os.getcwd()
#     cv2.imwrite(os.path.join(path, 'param_70_43_8.6_'+str(a[i])+'.png'), pic_a)
#     print('pictures saved:'+str(i))
=======
if __name__ == '__main__':
    saturn      = photo_to_sphere(img_saturn)
    print('Saturn image loaded.')
    gargantua   = photo_to_sphere(img_gargantua)
    print('Gargantua image loaded.')

    raytracer = wrmhole.wormhole_with_symmetry(steps=3000, initialcond = [20, np.pi, np.pi/2], Nz=1024, Ny=2048)
    # print(raytracer.shape)
    print('Ray tracer solution loaded.')
    print('Starting image placing process...')
    pic = decide_universe(raytracer, saturn, gargantua)
    print(pic)
    print('Image placing completed.')
    print('Saving picture')
    path = os.getcwd()
    cv2.imwrite(os.path.join(path, 'Pictures/Interstellar_met_grid4.png'), pic)
    print('Picture saved')
>>>>>>> 4f34cfe7a10d26db206a31354d4453fcdd16c1a5
