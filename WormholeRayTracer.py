import numpy as np
import cv2
import matplotlib.pyplot as plt
import RungeKutta as rk
import InbeddingDiagramDNeg as Dia
import Symplectic_DNeg as Smpl
import time
import os
#import scipy as sc


#Dit is op de master branch:


def dneg_r(y, M=0.43/1.42953 , p=1, a=0):
    # input: scalars
    # output: scalar
    # define r(l) for a DNeg wormhole without gravity

    x = 2*(np.abs(y) - a)/(np.pi*M)
    r = p + M*(x*np.arctan(x) - 0.5*np.log(1 + x**2))

    return r


def dneg_dr_dl(y, M=0.43/1.42953):
    # input:scalars
    # output: scalar
    # define derivative of r to l

    x = 2*np.abs(y)/(np.pi*M)
    dr_dl = 2/np.pi*np.arctan(x)

    return dr_dl


def dneg_d2r_dl2(y, M=0.43/1.42953):
    # input: scalars
    # output: scalars
    # define second derivative of r to l

    d2r_dl2 = 4*M*y/(np.pi**2*M**2*np.abs(y) + 4*np.abs(y)**3)

    return d2r_dl2


def screen_cart(Nz, Ny, L = 1):
     # input: Nz amount of pixels on vertical side screen
     #        Ny amount pixels horizontal side screen ,
     #        L = physical width and lenght of the screen.
     # output: 3D matrix (2d matrix of each ray/pixel, containing its location in 3D space)

    My = np.linspace(-L/2, L/2, Ny)
    Mz = np.linspace(-L/2, L/2, Nz)

    #cartesian product My X Mz
    arr = []
    for j in range(Nz):
        for i in range(Ny):
            # Placed at x = 1, (y,z) in My X Mz
            arr.append([1, My[i],Mz[j]]) #(x, y, z)

    return np.array(arr).reshape(Nz, Ny, 3) #Flat array into matrix


def cart_Sph(v):
    # input: matrix with cart. coord on first row,
    # output: matrix with Sph. coord on first row

    x,y,z = v

    # from carthesian to spherical coordinates
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    v_sph = np.array([r, phi, theta])

    return v_sph


def inn_momenta(S_c, S_sph, Cst_f, inn_p_f):
    # input: S_c: 3D matrix as an output of "screen_cart",
    #        S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
    #               within that containing value for that coordinate
    #        Cst_f: function that calculates constant of motion
    #        inn_p_f: function that calculates inn. momenta
    # output: p: 3D matrix with coordinates in impulse space on first row and
    #         then a 2D matrix within that with the value for each ray,
    #         Cst: list of cst of motion containing the value for each ray in 2D matrix

    r, phi, theta = S_sph
    S_n = S_c/r.reshape(len(S_c) ,len(S_c[0]), 1) # normalize direction light rays
    S_n = np.transpose(S_n, (2,0,1)) # start array in terms of coordinates
    p = inn_p_f(S_n, S_sph) # calculate initial momenta, coords still on first row matrix
    Cst = Cst_f(p, S_sph) # calculate constant of motions

    return [p, Cst]


def Cst_DNeg(p, q):
    # input: p: matrix with coordinates in momentum space on first row,
    #        q: matrix with coordinates in configuration space on first row ,
    # output: list of cst of motion containing the value for each ray in 2D matrix

    p_l, p_phi, p_th = p
    l, phi, theta = q

    # defining the constants of motion
    b = p_phi
    B_2 = p_th**2 + p_phi**2/np.sin(theta)**2
    Cst = np.array([b, B_2])

    return Cst


def inn_mom_DNeg(S_n, S_sph):
    # input: S_c: 3D matrix as earlier defined the in output of "screen_cart", from which
    #             we can calculate S_n
    #        S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
    #               within that containing value for that coordinate,
    # output: 3D matrix with coordinates in impulse space on first row and then
    #         a 2D matrix within that with the value for each ray

    l, phi, theta = S_sph

    # defining r(l)
    r = dneg_r(l)

    # defining the momenta
    p_l = -S_n[0]
    p_phi = -r*np.sin(theta)*S_n[1]
    p_th = r*S_n[2]
    p = np.array([p_l, p_phi, p_th])

    return p


def Simulate_DNeg(integrator, h, N, q0, Nz = 14**2, Ny = 14**2):
    #input: function that integrates(p(t), q(t)) to (p(t + h), q(t + h))
    #       h: stepsize
    #       N amount of steps
    #       Ni pixels
    #       loc: initial position
    #output: motion: 5D matrix the elements being [p, q] p, q being 3D matrices
    #        pict: 3D matrix (2D grid containg value in colorspace)

    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    q1 = np.transpose(np.tile(q0, (400,400,1)), (2,0,1)) + h*0.1
    q = q1
    Motion = [[p, q]]
    H = []

    start = time.time()

    # Integration
    for i in range(N):
        p, q , H_i = integrator(p, q, Cst, h)
        Motion.append([p, q])
        H.append(H_i)
    H.append(DNeg_Ham(p, q))

    end = time.time()

    print(end - start)

    pict = Make_Pict_RB(q, q1, 200, 0.5, 0.2)
    #print(pict)

    return np.array(Motion), pict , np.array(H)


def Make_Pict_RB(q, q0, N_a, R, w):
    # input: q: matrix with coordinates in configuration space on first row
    #        q0: starting position (unused)
    #        3D matrix (2D matrix of rays each containing a coordinate in colorspace)
    #        N_a: subdivision angles
    #        N_r: linspace radius to form grid
    #        h: width lines grid

    Par_phi = np.linspace(0, 2*np.pi, N_a-1)
    Par_th = np.linspace(0, np.pi, N_a-1)
    pict = []

    for j in range(len(q[0])):
        row = []

        for i in range(len(q[0][0])):
            r = q[0][j,i]
            phi = q[1][j,i]
            th = q[2][j,i]
            # Defines point on spherical grid
            on_shell = (np.abs(R - np.mod(r, R)) < R*w) or (np.abs(np.mod(r, R) - R) < R*w)
            on_phi = np.any(np.abs(phi - Par_phi) < 2*np.pi/N_a*w)
            on_theta = np.any(np.abs(th - Par_th) < np.pi/N_a*w)
            # Boolean conditions for when rays lands on spherical grid
            if (on_phi and on_theta) or (on_phi and on_shell) or (on_shell and on_theta):
                row.append(np.array([0, 0, 0]))

            else:
                # colors based on sign azimutha angle and inclination 
                if phi > np.pi and th > np.pi/2:
                    row.append([0, 1, 0])
                elif phi > np.pi and th < np.pi/2:
                    row.append([1, 0, 0])
                elif phi < np.pi and th > np.pi/2:
                    row.append([0, 0, 1])
                elif phi < np.pi and th < np.pi/2:
                    row.append([0.5, 0.5, 0])

            if r < 0 and np.linalg.norm(row[-1]) != 0:
                # invert color for points on oposite side of wormhole
                row[-1] = [(1 - row[-1][k]) for k in range(3)]

        pict.append(np.array(row))

    pict = cv2.cvtColor(np.array(pict, np.float32), 1)

    # pict = Image.fromarray(np.array(pict), 'RGB')

    return pict


def sum_subd(A):
    # input: A: 2D matrix such that the length of sides have int squares
    # sums subdivisions of array
    Ny, Nz =  A.shape
    # subdivides by making blocks with the square of the original lenght as size
    Ny_s = int(np.sqrt(Ny))
    Nz_s = int(np.sqrt(Nz))
    B = np.zeros((Ny_s, Nz_s))

    for i in range(Ny_s):
        for j in range(Nz_s):
            B[i,j] = np.sum(A[Ny_s*i:Ny_s*(i+1), Nz_s*j:Nz_s*(j+1)])

    return B


def DNeg_Ham(p, q , M = 0.43/1.42953, rho = 1):
    #input: p, q  3D matrices as defined earlier
    #output: 1D matrix, hamiltonian defined in each timestep

    p_l, p_phi, p_th = p
    l, phi, theta = q

    # defining r(l):
    r = dneg_r(l, M, rho)

    rec_r = 1/r
    rec_r_2 = rec_r**2
    sin1 = np.sin(theta)
    sin2 = sin1**2

    # defining hamiltonian
    H1 = p_l**2
    H2 = p_th**2*rec_r_2
    H3 = p_phi**2/sin2*rec_r_2

    return 0.5*sum_subd((H1 + H2 + H3))


def plot_Ham(H):
    #input: 3D array containing energy of each ray over time, advancement in time on first row
    # plot de hamiltoniaan van de partities van de rays
    Ny, Nz =  H[0].shape
    cl, ind = ray_spread(Ny, Nz)

    fig, ax = plt.subplots()
    x = np.arange(len(H))
    for i in range(Ny):
        for j in range(Nz):
            ij = i + Ny*j
            cl_i =cl[ind[ij]]
            ax.plot(x, H[:,i,j], color=cl_i)
    ax.set_yscale("log")
    ax.set_title("Donker pixels binnenkant scherm, lichte pixels buitenkant")
    plt.tight_layout()
    plt.show()


def ray_spread(Ny, Nz):
    # input: Ny: amount of horizontal arrays, Nz: amount of vertical arrays
    # output: cl: color based on deviation of the norm of a ray compared to direction obeserver is facing
            #ind ind of original ray mapped to colormap
    S_c = screen_cart(Ny, Nz)
    S_cT = np.transpose(S_c, (2,0,1))
    n = np.linalg.norm(S_cT, axis=0)
    n_u, ind = np. unique(n, return_inverse=True)
    N = n_u.size
    cl = plt.cm.viridis(np.arange(N)/N)

    return cl, ind


def gdsc(Motion):
    # input: Motion: 5D matrix, the elements being [p, q] with p, q as defined earlier

    Motion = np.transpose(Motion, (1,2,0,3,4))

    Ny, Nz =  Motion[0][0][0].shape
    Ny_s = int(np.sqrt(Ny))
    Nz_s = int(np.sqrt(Nz))
    # Samples a uniform portion of the rays for visualisation
    Sample = Motion[:, :, :, 1::Ny_s, 1::Nz_s]
    cl, ind = ray_spread(Ny_s, Nz_s)

    p, q = Sample
    p_l, p_phi, p_th = p
    l, phi, theta = q
    # caluclates coordinates in inbedded space
    ax = plt.figure().add_subplot(projection='3d')
    X, Y = dneg_r(l)*np.cos(phi), dneg_r(l)*np.sin(phi)
    Z = -Dia.imb_f_int(l)

    for i in range(Ny_s):
        for j in range(Nz_s):
            ij = i + Ny_s*j
            cl_i =cl[ind[ij]]
            ax.plot(X[:,i,j], Y[:,i,j], Z[:,i,j], color = cl_i, alpha=0.5)

    ax.set_title("Donker pixels binnenkant scherm, lichte pixels buitenkant")
    # adds surface
    Dia.inb_diagr([-10, 10], 1000, ax)
    plt.show()


#initial position in spherical coord
Motion1, Photo1, H1 = Simulate_DNeg(Smpl.Sympl_DNeg, 0.01, 1500, np.array([5, 3, 2]), 400, 400)
#Motion2, Photo2, H2 = Simulate_DNeg(rk.runge_kutta, 0.01, 1000, 9, 400, 400)

plot_Ham(H1)
#plot_Ham(H2)

gdsc(Motion1)
#gdsc(Motion2)

path = os.getcwd()
cv2.imwrite(path + '/DNeg Sympl.png', 255*Photo1)
#cv2.imwrite(path + '/DNeg Kutta.png', 255*Photo2)
