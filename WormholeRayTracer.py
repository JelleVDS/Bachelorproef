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
def dneg_r(y, M=0.43/1.42953 , rho=1, a=0):
    # input: scalars
    # output: scalar
    # define r(l) for a DNeg wormhole without gravity

    x = 2*(np.abs(y) - a)/(np.pi*M)
    r = rho + M*(x*np.arctan2(2*(np.abs(y) - a), np.pi*M) - 0.5*np.log(1 + x**2))

    return r

def dneg_dr_dl(y, M=0.43/1.42953, a=0):
    # input:scalars
    # output: scalar
    # define derivative of r to l

    x = 2*(np.abs(y)-a)/(np.pi*M)
    dr_dl = (2/np.pi)*np.arctan(x)*np.sign(y)

    return dr_dl


def dneg_d2r_dl2(y, M=0.43/1.42953, a=0):
    # input: scalars
    # output: scalars
    # define second derivative of r to l

    d2r_dl2 = (4*M)/(4*a**2 + M**2*np.pi**2 + 4*y**2 - 8*a*np.abs(y))

    return d2r_dl2


def screen_cart(Nz, Ny, L1 = 1, L2=2):
     # input: Nz amount of pixels on vertical side screen
     #        Ny amount pixels horizontal side screen ,
     #        L = physical width and lenght of the screen.
     # output: 3D matrix (2d matrix of each ray/pixel, containing its location in 3D space)

    My = np.linspace(-L2/2, L2/2, Ny)
    Mz = np.linspace(-L1/2, L1/2, Nz)

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


def Simulate_DNeg(integrator, Par, h, N, q0, Nz = 14**2, Ny = 14**2, Gr_D = '2D', mode = 0):
    #input: function that integrates(p(t), q(t)) to (p(t + h), q(t + h))
    #       h: stepsize
    #       N amount of steps
    #       Ni pixels
    #       q0: initial position
    #       mode: disable data collection
    #output: motion: 5D matrix the elements being [p, q] p, q being 3D matrices
    #        pict: 3D matrix (2D grid containg value in colorspace)

    S_c = screen_cart(Nz, Ny, 1, 1)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    q1 = np.transpose(np.tile(q0, (Nz, Ny,1)), (2,0,1)) + h*0.1
    q = q1
    Motion = [[p, q]]
    CM = []
    Grid = np.zeros((Nz, Ny), dtype=bool)

    start = time.time()

    # Integration
    for i in range(N):
        p, q , CM_i = integrator(p, q, Cst, h, Par)
        if mode == 0:
            Motion.append([p, q])
            CM.append(CM_i)
        if Gr_D == '3D':
            # change parameters grid here
            Grid = Grid_constr_3D(q, 11, 1, 0.01, Grid)

    if mode == 0:
        CM.append(DNeg_CM(p, q, Par))
        Motion.append([p, q])
    end = time.time()

    print(end - start)

    #pict = Make_Pict_RB(q)
    if Gr_D == '2D':
        Grid = Grid_constr_2D(q, 11, 1, 0.05)
    pict =  Make_Pict_RGBP(q, Grid)
    #print(pict)
    # pict = 0
    return np.array(Motion), pict, np.array(CM)


def diff_equations(t, variables):
    l, phi, theta, p_l, p_phi, p_th, M, rho, a = variables
    r = dneg_r(l, M, rho, a)
    rec_r = 1/r
    rec_r_2 = rec_r**2
    rec_r_3 = rec_r_2*rec_r
    rec_sin1 = 1/np.sin(theta)
    cos1 = np.cos(theta)
    rec_sin2 = rec_sin1**2
    rec_sin3 = rec_sin1*rec_sin2
    B = p_th**2 + p_phi**2 * rec_sin2
    b = p_phi

    # Using the hamiltonian equations of motion
    dl_dt       = p_l
    dtheta_dt   = p_th * rec_r_2

    dphi_dt     = b * rec_sin2 * rec_r_2
    dpl_dt      = B**2 * (dneg_dr_dl(l)) * rec_r_3
    dpth_dt     = b ** 2 * cos1 * rec_sin3 * rec_r_2

    diffeq = [-dl_dt, -dphi_dt, -dtheta_dt, -dpl_dt, np.zeros(dl_dt.shape), -dpth_dt, 0, 0, 0]
    return diffeq


def simulate_radius(t_end, Par, q0, Nz = 14**2, Ny = 14**2, methode = 'RK45'):
    """
    Solves the differential equations using a build in solver (solve_ivp) with
    specified method.
    Input:  - methode: method used for solving the ivp (standerd runge-kutta of fourth order)
            - h: stepsize
            - N: number of steps
            - Par: parameters wormhole
            - q0: initial position of the camera
            - Nz: number of vertical pixels
            - Ny: number of horizontal pixels
    Output: - sol.y: a matrix with each row containing the initial and end value
              for one ray: [l, phi, theta, p_l, p_phi, p_theta]
    """
    print('Initializing screen and calculating initial condition...')
    end = int(np.ceil(np.sqrt(Ny**2+Nz**2)))
    S_c = screen_cart(end, end)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    M, rho, a = Par
    p1, p2, p3 = p
    q1, q2, q3 = q0
    endpos = []
    endmom = []
    teller1 = int(len(p1)/2)
    # for teller1 in range(0, len(p1)):
    #     row_pos = []
    #     row_mom = []
    for teller2 in range(int(len(p1[0])/2), len(p1[0])):

        # start_it = time.time()
        initial_values = np.array([q1, q2, q3, p1[teller1][teller2], p2[teller1][teller2], p3[teller1][teller2], M, rho, a])
        sol            = integr.solve_ivp(diff_equations, [t_end, 0], initial_values, method = methode, t_eval=[0])
        #Reads out the data from the solution
        l_end       = sol.y[0][-1]
        phi_end     = sol.y[1][-1]
        # while phi_end>2*np.pi:
        #     phi_end = phi_end - 2*np.pi
        # while phi_end<0:
        #     phi_end = phi_end + 2*np.pi
        theta_end   = sol.y[2][-1]
        # while theta_end > np.pi:
        #     theta_end = theta_end - np.pi
        # while theta_end < 0:
        #     theta_end = theta_end + np.pi
        pl_end      = sol.y[3][-1]
        pphi_end    = sol.y[4][-1]
        ptheta_end  = sol.y[5][-1]
        # # adds local solution to row
        #     row_pos.append(np.array([l_end, phi_end, theta_end]))
        #     row_mom.append(np.array([pl_end, pphi_end, ptheta_end]))
        # end_it = time.time()
        # duration = end_it - start_it
        # print('Iteration ' + str((teller1, teller2)) + ' completed in ' + str(duration) + 's.')
        # adds row to matrix
        endpos.append(np.array([l_end, phi_end, theta_end]))
        endmom.append(np.array([pl_end, pphi_end, ptheta_end]))
    return np.array(endmom), np.array(endpos)

def simulate_raytracer(t_end, Par, q0, Nz = 14**2, Ny = 14**2, methode = 'RK45'):
    """
    Solves the differential equations using a build in solver (solve_ivp) with
    specified method.
    Input:  - methode: method used for solving the ivp (standerd runge-kutta of fourth order)
            - h: stepsize
            - N: number of steps
            - q0: initial position of the camera
            - Nz: number of vertical pixels
            - Ny: number of horizontal pixels
    Output: - sol.y: a matrix with each row containing the initial and end value
              for one ray: [l, phi, theta, p_l, p_phi, p_theta]
    """
    print('Initializing screen and calculating initial condition...')
    # end = int(np.ceil(np.sqrt(Ny**2+Nz**2)))
    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    p1, p2, p3 = p
    q1, q2, q3 = q0
    M, rho, a = Par
    endpos = []
    endmom = []
    print(len(p1))
    print(len(p1[0]))

    for teller1 in range(0, len(p1)):
        row_pos = []
        row_mom = []
        start_it = time.time()
        for teller2 in range(0, len(p1[0])):

            start_it = time.time()
            initial_values = np.array([q1, q2, q3, p1[teller1][teller2], p2[teller1][teller2], p3[teller1][teller2], M, rho, a])
            sol            = integr.solve_ivp(diff_equations, [t_end, 0], initial_values, method = methode, t_eval=[0])
            #Reads out the data from the solution
            l_end       = sol.y[0][-1]
            phi_end     = sol.y[1][-1]
            while phi_end>2*np.pi:
                phi_end = phi_end - 2*np.pi
            while phi_end<0:
                phi_end = phi_end + 2*np.pi
            theta_end   = sol.y[2][-1]
            while theta_end > np.pi:
                theta_end = theta_end - np.pi
            while theta_end < 0:
                theta_end = theta_end + np.pi
            pl_end      = sol.y[3][-1]
            pphi_end    = sol.y[4][-1]
            ptheta_end  = sol.y[5][-1]
            # adds local solution to row
            row_pos.append(np.array([l_end, phi_end, theta_end]))
            row_mom.append(np.array([pl_end, pphi_end, ptheta_end]))

        # adds row to matrix
        endpos.append(np.array(row_pos))
        endmom.append(np.array(row_mom))
        end_it = time.time()
        duration = end_it - start_it
        print('Iteration ' + str((teller1, teller2)) + ' completed in ' + str(duration) + 's.')
    return np.array(endmom), np.array(endpos)


def rotate_ray(ray, Nz, Ny):
    """
    The function assumes a 'horizontal' ray for theta = pi/2 and phi: pi to 2pi
    with position values and returns a full 2D picture of the wormhole.
    Inputs: - ray: the calculated 1D line
            - Nz: vertical number of pixels
            - Ny: horizontal number of pixels
    Output: - pic: a 5D array with the pixels and their l, phi, theta
    """
    Mz = np.arange(-Nz/2, Nz/2, 1)
    My = np.arange(-Ny/2, Ny/2, 1)
    pic = np.zeros((Nz, Ny, 3))
    print(pic.shape)
    for height in Mz:
        for width in My:
            r = int(round(np.sqrt(width**2 + height**2)))
            if r == len(ray):
                r = r-1
            z = int(height + Nz/2)
            y = int(width + Ny/2)
            l, phi, theta = ray[r]

            phi_rot     = width*(2*np.pi/Nz)
            theta_rot   = height*(np.pi/Ny)

            phi = phi + phi_rot
            theta = theta + theta_rot
            while phi>2*np.pi:
                phi = phi - 2*np.pi
            while phi<0:
                phi = phi + 2*np.pi
            while theta > np.pi:
                theta = theta - np.pi
            while theta < 0:
                theta = theta + np.pi
            loc = np.array([l, phi, theta])
            pic[z][y] = loc

    return pic



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


def Grid_constr_2D(q, N_a, R, w):
    # input: q: matrix with coordinates in configuration space on first row
    #        N_a: subdivision angles
    #        N_r: linspace radius to form grid
    #        w: ratio
    #output: 2D boolean array
    Nz, Ny =  q[0].shape
    r, phi, theta = q

    Par_phi = np.linspace(0, 2*np.pi, N_a-1)
    Par_th = np.linspace(0, np.pi, N_a-1)

    on_phi = np.any(
        np.abs(phi.reshape(1,Nz,Ny) -
               np.transpose(np.tile(Par_phi, (Nz, Ny,1)), (2,0,1)))
        < 2*np.pi/N_a*w, axis=0)

    on_theta = np.any(
        np.abs(theta.reshape(1,Nz,Ny) -
               np.transpose(np.tile(Par_th, (Nz, Ny,1)), (2,0,1)))
        < np.pi/N_a*w,  axis=0)

    return on_phi | on_theta


def Grid_constr_3D(q, N_a, R, w, Slice = None):
    # input: q: matrix with coordinates in configuration space on first row
    #        N_a: subdivision angles
    #        N_r: linspace radius to form grid
    #        w: ratio
    #output: 2D boolean array
    Nz, Ny =  q[0].shape
    if np.any(Slice == None):
        Slice = np.zeros((Nz, Ny), dtype=bool)
    Slice_inv = ~Slice
    r, phi, theta = q

    Par_phi = np.linspace(0, 2*np.pi, N_a-1)
    Par_th = np.linspace(0, np.pi, N_a-1)

    # Defines point on spherical grid
    rr = r[Slice_inv]
    M = len(rr)
    on_shell = (np.abs(R - np.mod(rr, R)) < R*w) | (np.abs(np.mod(rr, R) - R) < R*w)

    on_phi = np.any(
        np.abs(phi[Slice_inv].reshape(1,M) - np.tile(Par_phi, (M,1)).T)
        < 2*np.pi/N_a*w, axis=0)

    on_theta = np.any(
        np.abs(theta[Slice_inv].reshape(1,M) - np.tile(Par_th, (M,1)).T)
        < np.pi/N_a*w,  axis=0)

    # Boolean conditions for when rays lands on spherical grid
    Slice[Slice_inv] = (on_phi & on_theta) | (on_phi & on_shell) | (on_shell & on_theta)
    return Slice


def Make_Pict_RGBP(q, Grid):
    # input: q: matrix with coordinates in configuration space on first row
    # output: 3D matrix (2D matrix of rays each containing a coordinate in colorspace)

    pict = []

    for j in range(len(q[0])):
        row = []

        for i in range(len(q[0][0])):
            if Grid[j,i] == True:
                row.append([0,0,0])
            else:
                r = q[0][j,i]
                phi = q[1][j,i]
                th = q[2][j,i]
                # colors based on sign azimutha angle and inclination
                if phi > np.pi and th > np.pi/2:
                    row.append([0, 1, 0])
                elif phi > np.pi and th < np.pi/2:
                    row.append([1, 0, 0])
                elif phi < np.pi and th > np.pi/2:
                    row.append([0, 0, 1])
                elif phi < np.pi and th < np.pi/2:
                    row.append([0.5, 0.5, 0])

                if r < 0:
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


def DNeg_CM(p, q , Par):
    #input: p, q  3D matrices as defined earlier
    #output: 1D matrix, constants of Motion defined in each timestep
    M, rho, a = Par

    p_l, p_phi, p_th = p
    l, phi, theta = q

    # defining r(l):
    r = dneg_r(l, M, rho, a)

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


def plot_CM(CM, Label, name, path):
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
        ax[k].set_title(Label[k] + ",  Donker pixels binnenkant scherm, lichte pixels buitenkant")
    plt.tight_layout()
    plt.savefig(os.path.join(path, name), dpi=150)
    #plt.show()


def ray_spread(Nz, Ny):
    # input: Ny: amount of horizontal arrays, Nz: amount of vertical arrays
    # output: cl: color based on deviation of the norm of a ray compared to direction obeserver is facing
            #ind ind of original ray mapped to colormap
    S_c = screen_cart(Nz, Ny, 1, 1)
    S_cT = np.transpose(S_c, (2,0,1))
    n = np.linalg.norm(S_cT, axis=0)
    n_u, ind = np. unique(n, return_inverse=True)
    N = n_u.size
    cl = plt.cm.viridis(np.arange(N)/N)

    return cl, ind


def gdsc(Motion, Par, name, path, select = None):
    # input: Motion: 5D matrix, the elements being [p, q] with p, q as defined earlier
    M, rho, a = Par

    if np.any(select == None):
        Motion = np.transpose(Motion, (1,2,0,3,4))

        Ny, Nz =  Motion[0][0][0].shape
        Ny_s = int(np.sqrt(Nz))
        Nz_s = int(np.sqrt(Ny))

        # Samples a uniform portion of the rays for visualisation
        Sample = Motion[:, :, :, 1::Nz_s, 1::Ny_s]
        cl, ind = ray_spread(Nz_s, Ny_s)
    else:
        Motion = np.transpose(Motion, (3,4,0,1,2))
        Sample = np.transpose(
            [Motion[tuple(select[k])] for k in range(len(select))]
            , (2,3,1,0))

    p, q = Sample
    p_l, p_phi, p_th = p
    l, phi, theta = q
    # caluclates coordinates in inbedded space
    ax = plt.figure().add_subplot(projection='3d')
    r = dneg_r(l, M, rho, a)
    X, Y = r*np.cos(phi), r*np.sin(phi)
    Z = Dia.imb_f_int(l, Par)

    if np.any(select == None):
        for i in range(Nz_s):
            for j in range(Ny_s):
                ij = i + Nz_s*j
                cl_i =cl[ind[ij]]
                ax.plot(X[:,i,j], Y[:,i,j], Z[:,i,j], color = cl_i, alpha=0.5)
        ax.set_title("Donker pixels binnenkant scherm, lichte pixels buitenkant")
    else:
        for k in range(len(select)):
            ax.plot(X[:,k], Y[:,k], Z[:,k], label= str(select[k]))
        ax.legend()
    # adds surface

    S_l = np.linspace(np.max(l), np.min(l), len(l)+1)
    S_phi = np.linspace(0, 2*np.pi, len(l))
    S_L, S_PHI = np.meshgrid(dneg_r(S_l, M, rho, a), S_phi) # radius is r(l)

    # tile want symmetrisch voor rotaties, onafhankelijk van phi
    # Integraal voor Z richting zoals gedefinieerd in de paper
    S_Z = np.tile(Dia.imb_f_int(S_l, Par), (len(l), 1)) #z(l)

    S_X, S_Y = S_L*np.cos(S_PHI), S_L*np.sin(S_PHI)
    ax.plot_surface(S_X, S_Y, S_Z, cmap=plt.cm.YlGnBu_r, alpha=0.5)
    plt.savefig(os.path.join(path, name), dpi=150)
    #plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    Par = [0.43/1.42953, 1, 0] # M, rho, a parameters wormhole
    Integrator = 0
    #initial position in spherical coord
    #for radius in range(1, 15):
# <<<<<<< HEAD

    if Integrator == 1:
        initial_q = np.array([[17, np.pi, np.pi/2]])
        Grid_dimension = '3D'
        mode = 1
        Motion1, Photo1, CM1 = Simulate_DNeg(Smpl.Sympl_DNeg, Par, 0.01, 3000, initial_q, 20**2, 20**2, Grid_dimension, mode)
        #np.save('raytracer1', np.transpose(Motion1[-1][1], (1, 2, 0)))
        # Motion2, Photo2, CM2 = Simulate_DNeg(rk.runge_kutta, 0.01, 1000, [7, np.pi, np.pi/2], 10, 10)
        if mode == 0:
            plot_CM(CM1, ['H', 'b', 'B**2'], "Pictures/CM DNeg Sympl"+str(initial_q)+".png", path)
         #plot_CM(CM2, ['H', 'b', 'B**2'])

    if Integrator == 0:
        Nz = 200
        Ny = 400
        start = time.time()
        sol = simulate_raytracer(22, Par, [20, np.pi, np.pi/2], Nz, Ny, methode = 'RK45')
        end = time.time()
        print('Tijdsduur = ' + str(end-start))
        momenta, position = sol

        picture = rotate_ray(position, Nz, Ny)
        print(position)
        print('saving location...')
        np.save('raytracer2', picture)
        print('location saved!')

        print('Saving picture')
        path = os.getcwd()
        cv2.imwrite(os.path.join(path, 'picture2.png'), picture)
        print('Picture saved')

    #print(picture)
    if Integrator == 1:
        if mode ==  0:
            Geo_Sel = None
            #Geo_Sel = [[377, 24], [54, 341], [200, 200], [86, 39], [390, 390]]
            gdsc(Motion1, Par, "Pictures/geodesics DNeg Sympl"+str(initial_q)+".png", path, Geo_Sel)
            #gdsc(Motion2)

        # cv2.imwrite(os.path.join(path, "Pictures/Image "+Grid_dimension+"Gr DNeg Sympl"+str(initial_q)+".png"), 255*Photo1)
        # Motion2, Photo2, CM2 = Simulate_DNeg(rk.runge_kutta, 0.01, 1000, 9, 20**2, 20**2)
        #np.save('ray_solved', Motion1[-1])
        # np.save('ray_solved', Motion1)
        if mode ==  0:
             plot_CM(CM1, ['H', 'b', 'B**2'], "Pictures/CM DNeg Sympl"+str(Par)+" "+str(initial_q)+".png", path)
        #plot_CM(CM2, ['H', 'b', 'B**2'])

        #start = time.time()
        #sol = simulate_raytracer(0.01, 100, [5, 3, 3], Nz = 20**2, Ny = 20**2, methode = 'RK45')
        #end = time.time()
        #print('Tijdsduur = ' + str(end-start))
        #print(sol)
        #np.save('raytracer2', sol)`

        if mode ==  0:
             Geo_Sel = None
             #Geo_Sel = [[348, 70], [296, 360], [171, 175], [85, 37], [10, 10]]
             if Geo_Sel == None:
                 Geo_txt = ""
             else:
                 Geo_txt = str(Geo_Sel)
             gdsc(Motion1, Par, "Pictures/geodesics "+Geo_txt+" DNeg Sympl"+str(Par)+" "+str(initial_q)+".png", path, Geo_Sel)
        # # gdsc(Motion2)

        cv2.imwrite(os.path.join(path, "Pictures/Image "+Grid_dimension+"Gr DNeg Sympl"+str(Par)+" "+str(initial_q)+".png"), 255*Photo1)

        #cv2.imwrite(path + '/DNeg Kutta.png', 255*Photo2)
