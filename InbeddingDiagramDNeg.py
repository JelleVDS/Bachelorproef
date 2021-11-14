

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dneg_r(y, M=0.43/1.42953 , rho=8.6, a=4.3):
    # input: scalars
    # output: scalar
    # define r(l) for a DNeg wormhole without gravity

    x = 2*(np.abs(y) - a)/(np.pi*M)
    r = rho + M*(x*np.arctan(x) - 0.5*np.log(1 + x**2))

    return r


def dneg_dr_dl(y, M=0.43/1.42953, a=4.3):
    # input:scalars
    # output: scalar
    # define derivative of r to l

    x = 2*(np.abs(y) - a)/(np.pi*M)
    dr_dl = 2/np.pi*np.arctan(x)*np.sign(y)

    return dr_dl


def imb_f(l, Par):
    #input: 1D array
    #output: 1D array
    M, rho, a = Par
    #hoogte formule inbedding diagram
    h = np.sqrt(1 - dneg_dr_dl(l, M, a)**2)

    return h


def imb_f_int(l, Par):
    #input: 1D array
    #output: 1D array

    Z = []

    # integratie voor stijgende l
    for i in range(len(l)):
         Z.append(np.trapz(imb_f(l[:i], Par), l[:i], axis= 0))

    return np.array(Z)


def inb_diagr(I, N , Par, ax = None):
    #input: I: interval
    #       N: amount of points
    #output: plot
    M, rho, a = Par
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    l = np.linspace(I[0], I[1], N+1) # N+1, want dan N intervallen
    phi = np.linspace(0, 2*np.pi, N)
    L, PHI = np.meshgrid(dneg_r(l, M, rho, a), phi) # radius is r(l)

    # tile want symmetrisch voor rotaties, onafhankelijk van phi
    # Integraal voor Z richting zoals gedefinieerd in de paper
    Z = np.tile(imb_f_int(l, Par), (N, 1)) #z(l)

    X, Y = L*np.cos(PHI), L*np.sin(PHI)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r, alpha=0.5)

    if ax == None:
        plt.show()
        plt.savefig('EMB_W=0.43__rho=8.6__a=4.3.png')


if __name__ == '__main__':
    inb_diagr([-10, 10], 1000, Par = [0.43/1.42953, 8.6, 4.3])
