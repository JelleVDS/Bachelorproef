
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Dit is op de master branch
def dneg_r(y, M=0.43/1.42953 , p=1, a=0):
    # input: scalar, output: scalar
    # define r(l) for a DNeg wormhole without gravity
    x = 2*(np.abs(y) - a)/(np.pi*M)
    return p + M*(x*np.arctan(x) - 0.5*np.log(1 + x**2))

def dneg_dr_dl(y, M=0.43/1.42953):
    # input:scalar , output: scalar
    # define derivative of r to l
    x = 2*np.abs(y)/(np.pi*M)
    return 2/np.pi*np.arctan(x)

def dneg_d2r_dl2(y, M=0.43/1.42953):
    return 4*M*y/(np.pi**2*M**2*np.abs(y) + 4*np.abs(y)**3)

def screen_cart(Nz, Ny, L = 1):
     # input: Nz amount of pixels on vertical side screen, Ny amount pixels horizontal side screen ,
     # L = physical width and lenght of the screen. output: 3D matrix (2d matrix of each ray/pixel,
     # containing its location in 3D space)
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
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return np.array([r, phi, theta])

def inn_momenta(S_c, S_sph, Cst_f, inn_p_f):
    # input: S_c: 3D matrix as an output of "screen_cart",
    # S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
    # within that containing value for that coordinate, Cst_f: function that
    # calculates constant of motion, inn_p_f: function that calculates inn. momenta ,
    # output: p: 3D matrix with coordinates in impulse space on first row and
    # then a 2D matrix within that with the value for each ray, Cst: list of cst
    # of motion containing the value for each ray in 2D matrix
    r, phi, theta = S_sph
    S_n = S_c/r.reshape(len(S_c) ,len(S_c[0]), 1) # normalize direction light rays
    S_n = np.transpose(S_n, (2,0,1)) # start array in terms of coordinates
    p = inn_p_f(S_n, S_sph) # calculate initial momenta, coords still on first row matrix
    Cst = Cst_f(p, S_sph) # calculate constant of motions
    return [p, Cst]

def Cst_DNeg(p, q):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row ,
    # output: list of cst of motion containing the value for each ray in 2D matrix
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b = p_phi
    B_2 = p_th**2 + p_phi**2/np.sin(theta)**2
    return np.array([b, B_2])

def inn_mom_DNeg(S_n, S_sph):
    # input: S_c: 3D matrix as earlier defined the in output of "screen_cart",
    # S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
    # within that containing value for that coordinate, #output: 3D matrix
    # with coordinates in impulse space on first row and then a 2D matrix
    # within that with the value for each ray
    l, phi, theta = S_sph
    r = dneg_r(l)
    p_l = -S_n[0]
    p_phi = -r*np.sin(theta)*S_n[1]
    p_th = r*S_n[2]
    return np.array([p_l, p_phi, p_th])

def Sympl_DNeg(p, q, Cst, h, M = 0.43/1.42953, rho = 1):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row,
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, M: scalar, rho: scalar, output: list of coordinates in
    # configuration space containing 2D matrix with value for each ray
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b, B_2 = Cst
    r = dneg_r(l, M, rho)
    rec_r = 1/r
    rec_r_2 = rec_r**2
    rec_r_3 = rec_r_2*rec_r
    dr = dneg_dr_dl(l, M)
    d2r = dneg_d2r_dl2(l, M)
    sin1 = np.sin(theta)
    cos1 = np.cos(theta)
    sin2 = sin1**2
    sin3 = sin1*sin2
    
    l_h = p_l
    phi_h = b/sin1**2*rec_r_2
    theta_h = p_th*rec_r_2
    
    p_l_h = B_2*dr*rec_r_3
    p_th_h = b**2*cos1/sin3*rec_r_2
    
    l_h2 = 0.5*p_l_h
    phi_h2 = -phi_h*(p_l*dr*rec_r + p_th*cos1/sin1*rec_r_2)
    theta_h2 = 0.5*p_th_h*rec_r_2 - p_l*p_th*dr*rec_r_3
    
    c = 0.5*r*d2r - 1.5*dr**2
    p_l_h2 = p_l*(b*phi_h*rec_r_2 + theta_h**2)*c
    p_th_h2 = -p_l*p_th_h*dr*rec_r + 0.5*phi_h**2*p_th*(2*sin2 - 3)
    
    h_2 = h**2
    q[0] += l_h*h + l_h2*h_2
    q[1] += phi_h*h + phi_h2*h_2
    q[2] += theta_h*h + theta_h2*h_2
    
    p[0] += p_l_h*h + p_l_h2*h_2
    p[2] += p_th_h*h + p_th_h2*h_2
    return p, q

def Simulate_DNeg(integrator, h, N, Nz = 400, Ny = 400):
    #input: function that integrates(p(t), q(t)) to (p(t + h), q(t + h))
    #h: stepsize, N amount of steps, Ni pixels, 
    #output: motion: 5D matrix the elements being [p, q] p, q being 3D matrices
    #pict: 3D matrix (2D grid containg value in colorspace)
    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    q = np.zeros(p.shape) + h*0.1
    Motion = [[p, q]]

    for i in range(N): #Integration
        p, q = integrator(p, q, Cst, h)
        Motion.append([p, q])
    pict = Make_Pict_RB(q)
    return np.array(Motion), pict

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

def DNeg_Ham(Motion, M = 0.43/1.42953, rho = 1):
    #input: 5D matrix, the elements being [p, q] with p, q as defined earlier
    #output: 1D matrix, hamiltonian defined in each timestep
    Motion = np.transpose(Motion, (1,2,3,4,0))
    p, q = Motion
    p_l, p_phi, p_th = p
    l, phi, theta = q
    r = dneg_r(l, M, rho)
    rec_r = 1/r
    rec_r_2 = rec_r**2
    sin1 = np.sin(theta)
    sin2 = sin1**2
    
    H1 = p_l**2
    H2 = p_th**2*rec_r_2
    H3 = p_phi**2/sin2*rec_r_2
    return 0.5*np.sum((H1 + H2 + H3), axis=(0,1))


def plot_1D(y):
    fig, ax = plt.subplots()
    x = np.arange(len(y))
    ax.plot(x, y)
    plt.tight_layout()
    plt.show()
    
Motion, Photo = Simulate_DNeg(Sympl_DNeg, 0.01, 1000)

plot_1D(DNeg_Ham(Motion))
cv2.imshow('DNeg', Photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
