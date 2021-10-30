
import numpy as np
import cv2


def dneg_r(y, M=0.43/1.42953 , p=1, a=0):
    """
     input: scalar, output: scalar
     define r(l) for a DNeg wormhole without gravity
    """
    x = 2*(np.abs(y) - a)/(np.pi*M)
    return p + M*(x*np.arctan(x) - 0.5*np.log(1 + x**2))

def dneg_dr_dl(y, M=0.43/1.42953):
    """
     input:scalar , output: scalar
     define derivative of r to l
    """
    x = 2*np.abs(y)/(np.pi*M)
    return 2/np.pi*np.arctan(x)

def screen_cart(Nz, Ny, L = 1, d = 1):
    """
    input:  Nz amount of pixels on vertical side screen, Ny amount pixels horizontal side screen ,
            L = physical width and lenght of the screen.
    output: 3D matrix (2d matrix of each ray/pixel,
            containing its location in 3D space)
            d is the distance of the screen to the observer
     """
    My = np.linspace(-L/2, L/2, Ny)
    Mz = np.linspace(-L/2, L/2, Nz)
    #cartesian product My X Mz
    arr = []
    for j in range(Nz):
        for i in range(Ny):
            # Placed at x = 1, (y,z) in My X Mz
            arr.append([d, My[i],Mz[j]]) #(x, y, z)
    return np.array(arr).reshape(Nz, Ny, 3) #Flat array into matrix

def cart_Sph(v):
    """
    input:  matrix with cart. coord on first row,
    output: matrix with Sph. coord on first row
    """
    x,y,z = v
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return np.array([r, phi, theta])

def inn_momenta(S_c, S_sph, Cst_f, inn_p_f):
    """
    input:  S_c: 3D matrix as an output of "screen_cart",
            S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
            within that containing value for that coordinate,
            Cst_f: function that calculates constant of motion,
            inn_p_f: function that calculates inn. momenta ,
    output: p: 3D matrix with coordinates in impulse space on first row and
            then a 2D matrix within that with the value for each ray,
            Cst: list of cst of motion containing the value for each ray in 2D matrix
    """
    r, phi, theta = S_sph
    S_n = S_c/r.reshape(len(S_c) ,len(S_c[0]), 1) # normalize direction light rays
    S_n = np.transpose(S_n, (2,0,1)) # start array in terms of coordinates
    p = inn_p_f(S_n, S_sph) # calculate initial momenta, coords still on first row matrix
    Cst = Cst_f(p, S_sph) # calculate constant of motions
    return [p, Cst]

def Cst_DNeg(p, q):
    """
     input: p: matrix with coordinates in momentum space on first row,
            q: matrix with coordinates in configuration space on first row ,
    output: list of cst of motion containing the value for each ray in 2D matrix
    """
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b = p_phi
    B_2 = p_th**2 + p_phi**2/np.sin(theta)**2
    return np.array([b, B_2])

def inn_mom_DNeg(S_n, S_sph):
    """
    input:  S_c: 3D matrix as in output of "screen_cart",
            S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
            within that containing value for that coordinate,
    output: 3D matrix with coordinates in impulse space on first row and
            then a 2D matrix within that with the value for each ray
    """
    l, phi, theta = S_sph
    r = dneg_r(l)
    p_l = -S_n[0]
    p_phi = -r*np.sin(theta)*S_n[1]
    p_th = r*S_n[2]
    return np.array([p_l, p_phi, p_th])

def q_upd_DNeg(p, q, Cst, h, M = 0.43/1.42953, rho = 1):
    """
    input:  p: matrix with coordinates in momentum space on first row,
            q: matrix with coordinates in configuration space on first row,
            Cst: list of cst of motion containing the value for each ray in 2D matrix,
            h: stepsize, M: scalar, rho: scalar,
    output: list of coordinates in momentum space containing 2D matrix
            with value for each ray
    """
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b, B_2 = Cst
    h0p5 = 0.5*h

    rec_r_2 = 1/dneg_r(l, M, rho)**2
    theta_half = rec_r_2
    phi_half = 1/np.sin(theta)**2*rec_r_2
    l = l + h*p_l
    rec_r_2 = 1/dneg_r(l, M, rho)**2
    theta = theta + h0p5*p_th*(theta_half + rec_r_2)
    phi = phi + h0p5*b/(phi_half + 1/np.sin(theta)**2*rec_r_2)
    return [l, phi, theta]

def p_upd_DNeg(p, q, Cst, h, M = 0.43/1.42953, rho = 1):
    """
    input:  p: matrix with coordinates in momentum space on first row,
            q: matrix with coordinates in configuration space on first row,
            Cst: list of cst of motion containing the value for each ray in 2D matrix,
            h: stepsize,
            M: scalar,
            rho: scalar,
    output: list of coordinates in configuration space containing
            2D matrix with value for each ray
    """
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b, B_2 = Cst
    h0p5 = 0.5*h
    rec_r = 1/dneg_r(l, M, rho)
    rec_r_2 = rec_r**2
    rec_r_3 = rec_r_2*rec_r
    dr = dneg_dr_dl(l, M)

    p_l = p_l - h0p5*B_2*dr*rec_r_3
    p_th = p_th - h0p5*b**2*np.cos(theta)/np.sin(theta)**3*rec_r_2
    return [p_l, p_phi, p_th]

def Sympl_ord2(p, q, Cst, h):
    """
    input:  p: matrix with coordinates in momentum space on first row,
            q: matrix with coordinates in configuration space on first row, µ
            Cst: list of cst of motion containing the value for each ray in 2D matrix,
            h: stepsize
    output: p: list of coordinates in momentum space containing 2D matrix with
            value for each ray, q: list of coordinates in configuration space containing
            2D matrix with value for each ray
    """
    p = p_upd_DNeg(p, q, Cst, h,)
    q = q_upd_DNeg(p, q, Cst, h,)
    p = p_upd_DNeg(p, q, Cst, h,)
    return p, q

def Simulate_DNeg(integrator, h, N, Nz = 400, Ny = 400):
    """
    input:  function that integrates(p(t), q(t)) to (p(t + h), q(t + h))
            h: stepsize, N amount of steps, Ni pixels,
    output: final value of 3D p, q matrix
            pict: 3D matrix (2D grid containg value in colorspace)
    """
    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
<<<<<<< Updated upstream
    q = np.zeros(p.shape) + h*0.1

    for i in range(N): #Integration
        p, q = Sympl_ord2(p, q, Cst, h, p_upd_DNeg, q_upd_DNeg)
    pict = Make_Pict_RB(q)
    return [p, q], pict

def Make_Pict_RB(q):
    """
    input: q: matrix with coordinates in configuration space on first row ouput:
    3D matrix (2D matrix of rays each containing a coordinate in colorspace)
    RGB based on sign(l), q = (l, phi, theta)
    """
=======
    q1 = np.transpose(np.tile(q0, (Nz, Ny,1)), (2,0,1)) + h*0.1
    q = q1
    Motion = [[p, q]]
    H = []

    start = time.time()

    # Integration
    for i in range(N):
<<<<<<< Updated upstream
        p, q , H_i = integrator(p, q, Cst, h)
        # Motion.append([p, q])
        # H.append(H_i)
    # H.append(DNeg_Ham(p, q))

=======
        p, q , CM_i = integrator(p, q, Cst, h)
        # Motion.append([p, q])
        # CM.append(CM_i)
    # CM.append(DNeg_CM(p, q))
    Motion = q
>>>>>>> Stashed changes
    end = time.time()

    print(end - start)

    pict = Make_Pict_RB(q, q1, 200, 0.5, 0.2)
    #print(pict)

<<<<<<< Updated upstream
    return np.array(Motion), pict , np.array(H)


def Make_Pict_RB(q, q0, N_a, R, w):
=======
    return np.array(Motion), pict , np.array(CM)

def diff_equations(t, variables):
    l, theta, phi, p_l, p_th, p_phi = variables
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
    # H = 0.5*sum_subd((H1 + H2 + H3))
    B = p_th**2 + p_phi**2/sin2
    b = p_phi

    #Using the hamiltonian equations of motion
    dl_dt       = p_l
    dtheta_dt   = p_th * rec_r_2
    dphi_dt     = b / sin2 * rec_r_2
    dpl_dt      = B**2 * (dneg_dr_dl(l)) * rec_r_3
    dpth_dt     = b ** 2 * cos1 / sin3 * rec_r_2

    diffeq = [dl_dt, dphi_dt, dtheta_dt, dpl_dt, np.zeros(dl_dt.shape), dpth_dt]
    return diffeq


def simulate_raytracer(h, N, q0, Nz = 14**2, Ny = 14**2, methode = 'RK45'):
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
    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    p1, p2, p3 = p
    q1, q2, q3 = q0
    t_end = N*h
    endpos = []
    endmom = []

    for teller1 in range(0, len(p1)):
        row_pos = []
        row_mom = []
        for teller2 in range(0, len(p1[0])):
            initial_values = np.array([q1, q3, q2, p1[teller1][teller2], p2[teller1][teller2], p3[teller1][teller2]])
            sol         = integr.solve_ivp(diff_equations, [0, t_end], initial_values, method = methode, t_eval=[0, t_end])
            #Reads out the data from the solution
            l_end       = sol.y[0][-1]
            phi_end     = sol.y[1][-1]
            theta_end   = sol.y[2][-1]
            pl_end      = sol.y[3][-1]
            pphi_end    = sol.y[4][-1]
            ptheta_end  = sol.y[5][-1]
            # adds local solution to row
            row_pos.append([l_end, phi_end, theta_end])
            row_mom.append([pl_end, pphi_end, ptheta_end])
        # adds row to matrix
        endpos.append(row_pos)
        endmom.append(row_mom)
    return endpos, endmom


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
>>>>>>> Stashed changes
    # input: q: matrix with coordinates in configuration space on first row
    #        q0: starting position (unused)
    #        3D matrix (2D matrix of rays each containing a coordinate in colorspace)
    #        N_a: subdivision angles
    #        N_r: linspace radius to form grid
    #        h: width lines grid

    Par_phi = np.linspace(0, 2*np.pi, N_a-1)
    Par_th = np.linspace(0, np.pi, N_a-1)
>>>>>>> Stashed changes
    pict = []
    for j in range(len(q[0])):
        row = []
        for i in range(len(q[0][0])):
            if q[0][j,i] <= 0:
                row.append([255, 0, 0])
            else:
<<<<<<< Updated upstream
                row.append([0, 0, 255])
        pict.append(row)
    return cv2.cvtColor(np.array(pict, np.float32), 1)


Motion, Photo = Simulate_DNeg(0.5, 1000)

cv2.imshow('DNeg', Photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
=======
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

    H = 0.5*sum_subd((H1 + H2 + H3))

    return H


def plot_Ham(H):
    #input: 3D array containing energy of each ray over time, advancement in time on first row
    # plot de hamiltoniaan van de partities van de rays

    Ny, Nz =  H[0].shape
    cl, ind = ray_spread(Nz, Ny)

    fig, ax = plt.subplots()
    x = np.arange(len(H))
    for i in range(Nz):
        for j in range(Ny):
            ij = i + Nz*j
            cl_i =cl[ind[ij]]
            ax.plot(x, H[:,i,j], color=cl_i)
    ax.set_yscale("log")
    ax.set_title("Donker pixels binnenkant scherm, lichte pixels buitenkant")
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
<<<<<<< Updated upstream
# Motion1, Photo1, H1 = Simulate_DNeg(Smpl.Sympl_DNeg, 0.01, 1500, np.array([5, 3, 2]), 20**2, 20**2)
Motion2, Photo2, H2 = Simulate_DNeg(rk.runge_kutta, 0.01, 5000, np.array([5, 3, 2]), 20**3, 20**3)

# plot_Ham(H1)
#plot_Ham(H2)
=======
# Motion1, Photo1, CM1 = Simulate_DNeg(Smpl.Sympl_DNeg, 0.01, 1500, np.array([5, 3, 2]), 20**2, 20**2)
# Motion2, Photo2, CM2 = Simulate_DNeg(rk.runge_kutta, 0.01, 1000, 9, 20**2, 20**2)
# np.save('ray_solved', Motion1)
# plot_CM(CM1, ['H', 'b', 'B**2'])
# plot_CM(CM2, ['H', 'b', 'B**2'])
>>>>>>> Stashed changes

# gdsc(Motion1)
#gdsc(Motion2)

<<<<<<< Updated upstream
path = os.getcwd()
# print(path)
# cv2.imwrite(path + '/DNeg_Sympl.png', 255*Photo1)
cv2.imwrite(path + '/DNeg Kutta.png', 255*Photo2)
>>>>>>> Stashed changes
=======
# path = os.getcwd()
# cv2.imwrite(os.path.join(path, 'DNeg Sympl.png'), 255*Photo1)
# cv2.imwrite(path + '/DNeg Kutta.png', 255*Photo2)

print(simulate_raytracer(0.01, 200, [5, 3, 3], methode = 'RK45'))
>>>>>>> Stashed changes
