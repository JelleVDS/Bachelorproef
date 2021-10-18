
import numpy as np
import cv2


def dneg_r(y, M=0.43/1.42953 , p=1, a=0):
    # input: scalar, output: scalar
    # define r
    x = 2*(np.abs(y) - a)/(np.pi*M)
    return p + M*(x*np.arctan(x) - 0.5*np.log(1 + x**2))

def dneg_dr_dl(y, M=0.43/1.42953):
    #input:scalar , output: scalar
    # define derivative of r to l
    x = 2*np.abs(y)/(np.pi*M)
    return 2/np.pi*np.arctan(x)

def screen_cart(Nz, Ny, L = 1):
     # input: Nz amount of pixels on vertical side screen, Ny amount pixels horizontal side screen ,
     # L = physical width and lenght of the screen. output: 3D matrix (2d matrix of each ray/pixel,
     # containing its locatian in 3D space)
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
    # input: S_c: 3D matrix as in output of "screen_cart",
    # S_sph: 3D matrix with Sph. coord. on first row and then a 2D matrix
    # within that containing value for that coordinate, Cst_f: function that
    # calulates constant of motion, inn_p_f: function that calculates inn. momenta ,
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
    # input: S_c: 3D matrix as in output of "screen_cart",
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

def q_upd_DNeg(p, q, Cst, h, M = 0.43/1.42953, rho = 1):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row,
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, M: scalar, rho: scalar, output: list of coordinates in
    # momentum space containing 2D matrix with value for each ray
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
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row,
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, M: scalar, rho: scalar, output: list of coordinates in
    # configuration space containing 2D matrix with value for each ray
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

def Sympl_ord2(p, q, Cst, h, p_upd, q_upd):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row, µ
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, p_upd: funtion that updates momenta, q_upd: funtion that updates location,
    # output: p: list of coordinates in momentum space containing 2D matrix with
    # value for each ray, q: list of coordinates in configuration space containing
    # 2D matrix with value for each ray
    p = p_upd(p, q, Cst, h,)
    q = p_upd(p, q, Cst, h,)
    p = p_upd(p, q, Cst, h,)
    return p, q

def Simulate_DNeg(h, N, Nz = 400, Ny = 400 ):
    # h stepsize, N amount of steps, Ni pixels
    S_c = screen_cart(Nz, Ny)
    S_cT = np.transpose(S_c, (2,0,1))
    S_sph = cart_Sph(S_cT)
    p, Cst = inn_momenta(S_c, S_sph, Cst_DNeg, inn_mom_DNeg)
    q = np.zeros(p.shape) + h*0.1

    for i in range(N): #Integration
        p, q = Sympl_ord2(p, q, Cst, h, p_upd_DNeg, q_upd_DNeg)
    pict = Make_Pict_RB(q)
    return [p, q], pict

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


Motion, Photo = Simulate_DNeg(0.01, 1000)

cv2.imshow('DNeg', Photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
