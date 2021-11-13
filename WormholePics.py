import cv2
import numpy as np

# Inladen foto's
print('Reading in pictures...')
img_saturn    = cv2.imread('four400.png')
img_gargantua = cv2.imread('negfour400.png')


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


    S_X, S_Y = S_L*np.cos(S_PHI), S_L*np.sin(S_PHI)
    ax.plot_surface(S_X, S_Y, S_Z, cmap=plt.cm.YlGnBu_r, alpha=0.5)
    plt.savefig(os.path.join(path, name), dpi=150)
