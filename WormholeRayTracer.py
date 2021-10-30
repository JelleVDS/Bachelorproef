import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import RungeKutta as rk
import InbeddingDiagramDNeg as Dia
import Symplectic_DNeg as Smpl
import time
import os
import scipy.integrate as integr
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
    q1 = np.transpose(np.tile(q0, (Nz, Ny,1)), (2,0,1)) + h*0.1
    q = q1
    Motion = [[p, q]]
    CM = []

    start = time.time()

    # Integration
    for i in range(N):
        p, q , CM_i = integrator(p, q, Cst, h)
        Motion.append([p, q])
        CM.append(CM_i)
    CM.append(DNeg_CM(p, q))

    end = time.time()

    print(end - start)

    pict = Make_Pict_RB(q)
    #print(pict)

    return np.array(Motion), pict , np.array(CM)

def diff_equations(l, theta, phi, p_l, p_th, p_phi):
    r = dneg_r(l)
    rec_r = 1/r
    rec_r_2 = rec_r**2
    rec_r_3 = rec_r_2*rec_r
    sin1 = np.sin(theta)
    cos1 = np.cos(theta)
    sin2 = sin1**2
    sin3 = sin1*sin2
    H1 = p_l**2
    H2 = p_th**2*rec_r_2
    H3 = p_phi**2/sin2*rec_r_2
    H = 0.5*sum_subd((H1 + H2 + H3))
    B2_C = sum_subd(p_th**2 + p_phi**2/sin2)
    b_C = sum_subd(p_phi)

    #Using the hamiltonian equations of motion
    dl_dt       = p_l
    dtheta_dt   = p_th * rec_r_2
    dphi_dt     = b / sin2 * rec_r_2
    dpl_dt      = B**2 * (dneg_dr_dl(l)) * rec_r_3
    dpth_dt     = b ** 2 * cos1 / sin3 * rec_r_2

    diffeq = [dl_dt, dphi_dt, dtheta_dt, dpl_dt, np.zeros(dl_dt.shape), dpth_dt]
    return diffeq

def simulate_raytracer (h, N, q0, Nz = 14**2, Ny = 14**2, methode = 'RK45'):
    """
    Solves the differential equations using a build in solver (solve_ivp) with
    specified method.
    Input:  - methode: method used for solving the ivp (standerd runge-kutta of fourth order)
            - h: stepsize
            - N: number of steps
            - q0: initial position of the camera
            - Nz: number of vertical pixels
            - Ny: number of horizontal pixels

    Output:
    """
    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    p1, p2, p3 = p
    q1, q2, q3 = q0
    initial_values = [q1, q3, q2, p1, p3, p2]
    t_end = N*h

    sol = integr.solve_ivp(diff_equations, [0, t_end], initial_values, method = methode )

    return sol

def Make_Pict_RB(q):
    # input: q: matrix with coordinates in configuration space on first row ouput:
    # 3D matrix (2D matrix of rays each containing a coordinate in colorspace)
    # RGB based on sign(l), q = (l, phi, theta)
    pict = []
    for j in range(len(q[0])):
        row = []
        for i in range(len(q[0][0])):
            if q[0][j,i] <= 0:
                row.append([255, 0, 0])
            else:
                row.append([0, 0, 255])
        pict.append(row)
    return cv2.cvtColor(np.array(pict, np.float32), 1)


def Make_Pict_RGBP(q, q0, N_a, R, w):
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
                # invert color for points on oposite side of wormhol
                row[-1] = [(1 - row[-1][k]) for k in range(3)]

        pict.append(np.array(row))

    pict = cv2.cvtColor(np.array(pict, np.float32), 1)

    # pict = Image.fromarray(np.array(pict), 'RGB')

    return pict


def sum_subd(A):
    # input: A: 2D matrix such that the length of sides have int squares
    #sums subdivisions of array

    Nz, Ny =  A.shape

    # subdivides by making blocks with the square of the original lenght as size
    Ny_s = int(np.sqrt(Nz))
    Nz_s = int(np.sqrt(Ny))
    B = np.zeros((Nz_s, Ny_s))

    for i in range(Nz_s):
        for j in range(Ny_s):
            B[i,j] = np.sum(A[Nz_s*i:Nz_s*(i+1), Ny_s*j:Ny_s*(j+1)])

    return B


def DNeg_CM(p, q , M = 0.43/1.42953, rho = 1):
    #input: p, q  3D matrices as defined earlier
    #output: 1D matrix, constants of Motion defined in each timestep

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

    H = 0.5*sum_subd((H1 + H2 + H3))
    B2_C = sum_subd(p_th**2 + p_phi**2/sin2)
    b_C = sum_subd(p_phi)

    return [H, b_C, B2_C]


def plot_CM(CM, Name):
    #input: 3D array containing energy of each ray over time, advancement in time on first row
    # plot the constants of motion over the partition of the rays

    Ny, Nz =  CM[0,0].shape
    CM = np.transpose(CM, (1,0,2,3))
    N_C = len(CM)
    cl, ind = ray_spread(Nz, Ny)

    fig, ax = plt.subplots(N_C, 1)
    x = np.arange(len(CM[0]))
    for k in range(N_C):
        for i in range(Nz):
            for j in range(Ny):
                ij = i + Nz*j
                cl_i =cl[ind[ij]]
                ax[k].plot(x, CM[k,:,i,j], color=cl_i)
        ax[k].set_yscale("log")
        ax[k].set_title(Name[k] + ",  Donker pixels binnenkant scherm, lichte pixels buitenkant")
    plt.tight_layout()
    plt.show()


def ray_spread(Nz, Ny):
    # input: Ny: amount of horizontal arrays, Nz: amount of vertical arrays
    # output: cl: color based on deviation of the norm of a ray compared to direction obeserver is facing
            #ind ind of original ray mapped to colormap
    S_c = screen_cart(Nz, Ny)
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
    Ny_s = int(np.sqrt(Nz))
    Nz_s = int(np.sqrt(Ny))

    # Samples a uniform portion of the rays for visualisation
    Sample = Motion[:, :, :, 1::Nz_s, 1::Ny_s]
    cl, ind = ray_spread(Nz_s, Ny_s)

    p, q = Sample
    p_l, p_phi, p_th = p
    l, phi, theta = q
    # caluclates coordinates in inbedded space
    ax = plt.figure().add_subplot(projection='3d')
    X, Y = dneg_r(l)*np.cos(phi), dneg_r(l)*np.sin(phi)
    Z = Dia.imb_f_int(l)

    for i in range(Nz_s):
        for j in range(Ny_s):
            ij = i + Nz_s*j
            cl_i =cl[ind[ij]]
            ax.plot(X[:,i,j], Y[:,i,j], Z[:,i,j], color = cl_i, alpha=0.5)
    # adds surface
    ax.set_title("Donker pixels binnenkant scherm, lichte pixels buitenkant")
    Dia.inb_diagr([-10, 10], 1000, ax)
    plt.show()


#initial position in spherical coord
Motion1, Photo1, CM1 = Simulate_DNeg(Smpl.Sympl_DNeg, 0.01, 1500, np.array([5, 3, 2]), 20**2, 20**2)
# Motion2, Photo2, CM2 = Simulate_DNeg(rk.runge_kutta, 0.01, 1000, 9, 20**2, 20**2)
np.save('ray_solved', Photo1)
# plot_CM(CM1, ['H', 'b', 'B**2'])
# plot_CM(CM2, ['H', 'b', 'B**2'])

# gdsc(Motion1)
# gdsc(Motion2)

# path = os.getcwd()
# cv2.imwrite(os.path.join(path, 'DNeg Sympl.png'), 255*Photo1)
# cv2.imwrite(path + '/DNeg Kutta.png', 255*Photo2)
