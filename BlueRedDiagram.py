import numpy as np
# import cv2
import matplotlib.pyplot as plt

"""
We pakken een scherm in carthesische coordinaten en zetten het om naar sferisch (zie wormhole)
Dan adhv van phi en theta waarden geven we een blauw/rood waarde aan elke pixel
theta = 0-> pi dus stappen van 255/pi
phi = 0 -> 2pi dus stappen van 255/(2pi)
"""

def screen_cart(Nz, Ny, L = 1):
     # input: Nz amount of pixels on vertical side screen, Ny amount pixels horizontal side screen ,
     # L = physical width and lenght of the screen. output: 3D matrix (2d matrix of each ray/pixel,
     # containing its location in 3D space)
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
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return (r, phi, theta)


def blue_red(Ny, Nz):
    # sferische coordinaten vinden van het scherm
    S_c   = screen_cart(Nz, Ny)
    S_cT  = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    r, phi, theta = S_sph

    # bereik van theta en phi van het scherm
    theta_max = np.max(theta[0])
    theta_min = np.min(theta[-1])
    phi_max = np.max(phi[0])
    bereik_theta = abs(theta_max - theta_min)
    bereik_phi   = 2 * phi_max

    # de blauw en roodwaarden voor theta en phi berekenen van het scherm
    blue = (255 / bereik_theta) * abs(theta - theta_max)
    red  = (255 / bereik_phi) * (phi + phi_max)
    return (blue, red)

print(blue_red(4, 6))
# print(cart_Sph([1, 1/2, 1/2]))
# print(cart_Sph([1, -1/2, -1/2]))
# S_c   = screen_cart(5, 6)
# S_cT  = np.transpose(S_c, (2,0,1))
# S_sph = cart_Sph(S_cT)
# r, phi, theta = S_sph
# print(S_cT)

def blue_red_image(Ny, Nz):
    blue, red = blue_red(Ny, Nz)
    pixels = []
    for i in range(0, Ny):
        row = []
        for j in range(0, Nz):
            row.append([red[j][i],0,blue[j][i]])
        pixels.append(row)
    return pixels

print(blue_red_image(4,6))

def make_image(Ny, Nz):
    pixels = blue_red_image(Ny, Nz)
    pic = cv2.cvtColor(np.array(pixels, np.float32), 1)
    return pic

def plot_1D(y):
    fig, ax = plt.subplots()
    x = np.arange(len(y))
    ax.plot(x, y)
    plt.tight_layout()
    plt.show()

plot_1D(make_image(400,400))
cv2.imshow('titel', Photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
