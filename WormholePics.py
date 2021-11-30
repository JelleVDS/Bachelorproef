from numba import njit
import cv2
import numpy as np
from math import floor

def read_in_pictures(sat, gar):
    """
    Reads in pictures for wormhole and determines the theta and
    phi values in lists.
    Input:  - sat: picture for the side where the camera is
            - gar: picture for the opposite side of the wormhole
    Output: - img_saturn: cv-form of the camera side picture
            - img_gargantua: cv-form of the other side picture
            - theta_list: list of all possible theta values
            - phi_list: list of all possible phi values
    """
    # Inladen foto's
    print('Reading in pictures...')
    img_saturn    = cv2.imread(sat)
    img_gargantua = cv2.imread(gar)

    #Maak lijsten om dichtste te zoeken
    vertical   = len(img_saturn)     #1024
    horizontal = len(img_saturn[0])  #2048

    theta_list = list()
    for teller in range(0, vertical):
        theta = (np.pi/vertical) * teller #- np.pi
        theta_list.append(theta)

    phi_list =list()
    for teller in range(0, horizontal):
        phi   = (2*np.pi/horizontal) * teller
        while phi>2*np.pi:
            phi = phi - 2*np.pi
        while phi<0:
            phi = phi + 2*np.pi
        phi_list.append(phi)

    return img_saturn, img_gargantua, theta_list, phi_list



def read_pics(saturn, gargantua):
    
    print('Reading in pictures...')
    img_saturn    = cv2.imread(saturn)
    img_gargantua = cv2.imread(gargantua)
    
    return img_saturn, img_gargantua
    

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
            phi   = (2*np.pi/horizontal) * column
            while phi>2*np.pi:
                phi = phi - 2*np.pi
            while phi<0:
                phi = phi + 2*np.pi
            coordinate = (theta, phi) #Tuple with angles that will be used as key
            pixel      = np.array([photo[row][column]]) #RGB-values
            dict[coordinate] = pixel

    return dict


def decide_universe(photo, saturn, gargantua, theta_list, phi_list):
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
                pixel = ray_to_rgb((photo[rij][kolom][1], photo[rij][kolom][2]), gargantua, theta_list, phi_list)
            else:
                pixel = ray_to_rgb((photo[rij][kolom][1], photo[rij][kolom][2]), saturn, theta_list, phi_list)

            [[R, G, B]] = pixel
            row.append([R, G, B])
        picture.append(np.array(row))
    # img = cv2.cvtColor(np.array(picture, np.float32), 1)
    return np.array(picture)

    # print('here 4')

@njit
def distance(x, position):
    """
    Define a distance function for closest neighbour
    """
    dist = abs(x-position)

    return dist


def determine_theta(Nz, theta):
    """
    Determines the location of the ray in the picture.
    Input:  - Nz: height of the picture in pixels
            - theta: the theta value of the ray
    Output: - i: row of the ray
    """
    i = int(floor(Nz * theta / np.pi))
    return i


def determine_phi(Ny, phi):
    """
    Determines the location of the ray in the picture.
    Input:  - Ny: height of the picture in pixels
            - phi: the phi value of the ray
    Output: - j: column of the ray
    """
    j = int(floor(Ny * phi / (2 * np.pi)))
    return j


def make_picture(photo, gargantua, saturn):
    Nz = len(photo)
    Ny = len(photo[0])
    
    pic = np.empty([Nz, Ny, 3])
    for rij in range(0, Nz):
        for kolom in range(0, Ny):
            element = photo[rij][kolom]
            l, phi, theta = element
            loctheta = determine_theta(Nz, theta)
            locphi = determine_phi(Ny, phi)
            if l < 0:
                pic[rij][kolom] = gargantua[loctheta][locphi]
            else:
                pic[rij][kolom] = saturn[loctheta][locphi]
    return pic


def make_pic_quick(pic, sat, gar):
    
    img_saturn, img_gargantua = read_pics(sat, gar)
    print('Pictures ready!')
    print('Making wormhole...')
    picture = make_picture(pic, img_saturn, img_gargantua)
    print('Wormhole ready!')
    
    return picture
    

def ray_to_rgb(position, saturn, theta_list, phi_list):
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


def make_wormhole_pic(pic, sat, gar):
    """
    Script to run wormhole picture as a whole.
    """
    img_saturn, img_gargantua, theta_list, phi_list = read_in_pictures(sat, gar)
    saturn = photo_to_sphere(img_saturn)
    gargantua = photo_to_sphere(img_gargantua)
    print('Pictures ready!')
    print('Making wormhole...')
    picture = decide_universe(pic, saturn, gargantua, theta_list, phi_list)
    print('Wormhole ready!')
    return picture
