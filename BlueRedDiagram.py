import numpy as np
import cv2
from PIL import Image


def screen_cart(Nz, Ny, L = 1):
     # input: Nz amount of pixels on vertical side screen,
     #        Ny amount pixels horizontal side screen ,
     #        L = physical width and lenght of the screen.
     # output: 3D matrix (2d matrix of each ray/pixel, containing its location in 3D space)
    My = np.linspace(-L, L, Ny)
    Mz = np.linspace(-L, L, Nz)

    #cartesian product My X Mz
    arr = []
    for j in range(Nz):
        for i in range(Ny):
            # Placed at x = 1, (y,z) in My X Mz
            arr.append([1, My[i]/2,Mz[j]/2]) #(x, y, z)
    return np.array(arr).reshape(Nz, Ny, 3) #Flat array into matrix

def cart_Sph(v):
    # input: matrix with cart. coord on first row,
    # output: matrix with Sph. coord on first row

    x,y,z = v

    # from carthesian to spherical coordinates
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    v_sph = (r, phi, theta)
    return v_sph


def blue_red(Ny, Nz):
    # input: Nz: the amount of pixels in the vertical side screen
    #        Ny: the amount of pixels in the horizontal side screen
    # output: (blue, red) where blue is an array of all the blue values (theta)
    #         and red is an array of all the red values (phi)

    # to find the spherical coordinates of the screen
    S_c   = screen_cart(Nz, Ny)
    S_cT  = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    r, phi, theta = S_sph

    # range of theta and phi coordinates of the screen
    theta_max = np.max(theta[0])
    theta_min = np.min(theta[-1])
    phi_max   = np.max(phi[0])
    bereik_theta = abs(theta_max - theta_min)
    bereik_phi   = 2 * phi_max

    # calculating the blue and red values (blue for theta and red for phi)
    # theta and phi coordinates of the screen
    # divided by 255!
    blue = ((np.around((255 / bereik_theta) * abs(theta - theta_max)))/255)
    red  = ((np.around((255 / bereik_phi) * (phi + phi_max)))/255)
    return (blue, red)

# nog weg doen later:
# print(blue_red(4, 6))
# print(cart_Sph([1, 1/2, 1/2]))
# print(cart_Sph([1, -1/2, -1/2]))
# S_c   = screen_cart(4, 6)
# S_cT  = np.transpose(S_c, (2,0,1))
# S_sph = cart_Sph(S_cT)
# r, phi, theta = S_sph
# print(np.min(theta[-1]))

def blue_red_image(Ny, Nz):
    # input: Nz: the amount of pixels in the vertical side screen
    #        Ny: the amount of pixels in the horizontal side screen
    # output: pixels: a matrix of the RGB values

    blue, red = blue_red(Ny, Nz)
    pixels = []
    for i in range(0, Ny):
        row = []
        for j in range(0, Nz):
            row.append([red[j][i],int(0),blue[j][i]])
        pixels.append(np.array(row))
    return np.array(pixels)


def make_image(Ny, Nz):
    # input: Nz: the amount of pixels in the vertical side screen
    #        Ny: the amount of pixels in the horizontal side screen
    # output: picture of the red-blue transition

    pixels = blue_red_image(Ny, Nz)
    # Later weg doen:
    # pic = Image.fromarray(pixels, 'RGB')
    # print(pixels)
    pic = cv2.cvtColor(np.array(pixels, np.float32), 1)
    return pic

Photo = make_image(400, 600)
# Later weg doen:
# Photo.show()
cv2.imshow('Blue-Red transition', Photo)
# Later weg doen:
# cv2.destroyAllWindows()
cv2.waitKey(0)
# Later weg doen:
# print(Photo)
