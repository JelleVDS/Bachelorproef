import numpy as np
from numba import njit
import time

def sum_subd(A):
    # A 2D/1D matrix such that the lengt of sides have int squares
    Sh = A.shape
    if len(Sh) > 1:
        Ny, Nz =  Sh
        Ny_s = int(np.sqrt(Ny))
        Nz_s = int(np.sqrt(Nz))
        B = np.zeros((Ny_s, Nz_s))
        for i in range(Ny_s):
            for j in range(Nz_s):
                B[i,j] = np.sum(A[Ny_s*i:Ny_s*(i+1), Nz_s*j:Nz_s*(j+1)])
    else:
        N = Sh
        N_s = int(np.sqrt(N))
        B = np.zeros(N_s)
        for i in range(N_s):
            B[i] = np.sum(A[N_s*i:N_s*(i+1)])
    return B
    

@njit
def Sympl_calc(P, Q, p, q, l, r, dr, d2r, Cst, h_vect, S):
    #b, B_2 = Cst
    p_l, p_phi, p_th = p
    phi, theta = q[1:]
    
    P[:,0] = p
    Q[:,0] = q
    
    rec_r = 1/r
    rec_r_2 = rec_r**2
    rec_r_3 = rec_r_2*rec_r
        
    sin1 = np.sin(theta)
    sin2 = sin1**2
    rec_sin1 = 1/sin1
    cos1 = np.cos(theta)
    rec_sin2 = rec_sin1**2
    rec_sin3 = rec_sin1*rec_sin2
    
    dr_2 = dr**2
    c = 0.5*r*d2r - 1.5*dr_2
    d = dr*rec_r
    w = p_l*d
    e = 0.5*p_l*np.sin(2*theta)*r*dr
    g = 2*sin2 - 3
    f = -p_th*g
        
    H1 = p_l**2
    H2 = p_th**2*rec_r_2
    H3 = p_phi**2*rec_sin2*rec_r_2
    H = 0.5*(H1 + H2 + H3)
    b = p_phi
    B_2 = p_th**2 + p_phi**2*rec_sin2
        
    Q[0,1] = p_l
    Q[1,1] = b*rec_sin2*rec_r_2
    Q[2,1] = p_th*rec_r_2
    
    P[0,1] = B_2*dr*rec_r_3
    P[2,1] = b**2*cos1*rec_sin3*rec_r_2
        
    m = Q[2,1]*cos1*rec_sin1
    phi1_2 = Q[1,1]**2
    y = p_th*w
    
    Q[0,2] = 0.5*P[0,1]
    Q[1,2] = -Q[1,1]*(w + m)
    Q[2,2] = (0.5*P[2,1] - y)*rec_r_2
    
    P[0,2] = p_l*rec_r_2**2*B_2*c
    P[2,2] = -phi1_2*(e + f)
        
    Q[0,3] = 0.5*P[0,2]
    Q[1,3] = 2*Q[1,1]*Q[2,1]*m*w
    Q[2,3] = -phi1_2*rec_r_2*(e + 0.5*f)
        
    s = phi1_2*g
    o = 0.5*s*Q[2,1]
        
    P[0,3] = o*d
    P[2,3] = -s*y
        
    Q[0,4] = 0.5*P[0,3]
    Q[2,4] = 0.75*P[2,3]*rec_r_2
    
    P[0,4] = p_l*o*Q[2,1]*(c - 2*dr_2)
        
    Q[0,5] = 0.5*P[0,4]
    
    P = np.sum(h_vect * np.transpose(P, S), axis=0)
    Q = np.sum(h_vect * np.transpose(Q, S), axis=0)
    Q[1] = np.mod(Q[1], 2*np.pi)
    Q[2] = np.mod(Q[2], np.pi)
    return P, Q, b, B_2, H

@njit
def Dneg_r_Br(l, l_abs, M, rho, a):  
    x = 2*(l_abs - a)/(np.pi*M)
    x_tan = np.arctan(x)
    r = rho + M*(x*x_tan - 0.5*np.log(1 + x**2))
    dr = 2/np.pi*x_tan*np.sign(l)
    d2r = (4*M)/(4*a**2 + M**2*np.pi**2 + 4*l**2 - 8*a*l_abs) 
    return r, dr, d2r

def Sympl_DNeg(p, q, Cst, h_vect, Par, P, Q, r, dr, d2r, S):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row,
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, M: scalar, rho: scalar, output: list of coordinates in
    # configuration space containing 2D matrix with value for each ray
    start = time.time()
    M, rho, a = Par
    
    l = q[0]
    l_abs = np.abs(l)
    l_con = l_abs >= a
    inv_l_con = ~l_con
    
    r[l_con], dr[l_con], d2r[l_con] =  Dneg_r_Br(l[l_con], l_abs[l_con], M, rho, a)
    r[inv_l_con] = rho
    dr[inv_l_con] = 0
    d2r[inv_l_con] = 0
    
    P, Q, b, B_2, H = Sympl_calc(P, Q, p, q, l, r, dr, d2r, Cst, h_vect, S)
    end = time.time()
    return (P, Q, [sum_subd(H), sum_subd(b), sum_subd(B_2)], end-start) 
    