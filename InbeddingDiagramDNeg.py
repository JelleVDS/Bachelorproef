

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dneg_r(l, M , rho, a):
    # input: scalars
    # output: scalar
    # define r(l) for a DNeg wormhole without gravity
    
    r = np.empty(l.shape)
    l_abs = np.abs(l)
    l_con = l_abs > a
    inv_l_con = ~l_con
    
    x = 2*(l_abs[l_con] - a)/(np.pi*M)
    r[l_con] = rho + M*(x*np.arctan2(2*(l_abs[l_con] - a), np.pi*M) - 0.5*np.log(1 + x**2))
    r[inv_l_con] = rho
    return r

def dneg_dr_dl(l, M, a):
    # input:scalars
    # output: scalar
    # define derivative of r to l
    
    dr_dl = np.empty(l.shape)
    l_abs = np.abs(l)
    l_con = l_abs > a
    inv_l_con = ~l_con
    
    x = 2*(l_abs[l_con] - a)/(np.pi*M)
    dr_dl[l_con] = (2/np.pi)*np.arctan(x)*np.sign(l[l_con])
    dr_dl[inv_l_con] = 0

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
    R, PHI = np.meshgrid(dneg_r(l, M, rho, a), phi) # radius is r(l)

    # tile want symmetrisch voor rotaties, onafhankelijk van phi
    # Integraal voor Z richting zoals gedefinieerd in de paper
    Z = np.tile(imb_f_int(l, Par), (N, 1)) #z(l)

    X, Y = R*np.cos(PHI), R*np.sin(PHI)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r, alpha=0.5)

    if ax == None:
        plt.show()
        plt.savefig('EMB_W=0.43__rho=8.6__a=4.3.png')


if __name__ == '__main__':
    inb_diagr([-10, 10], 1000, Par = [0.43/1.42953, 8.6, 4.3])
