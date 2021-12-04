from tqdm.auto import tqdm
import cv2
import numpy as np
from math import floor


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
    for rij in tqdm(range(0, Nz)):
        for kolom in range(0, Ny):
            element = photo[rij][kolom]
            l, phi, theta = element
            loctheta = determine_theta(len(saturn), theta)
            locphi = determine_phi(len(saturn[0]), phi)
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
