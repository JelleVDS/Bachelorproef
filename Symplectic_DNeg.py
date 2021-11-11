import numpy as np

def sum_subd(A):
    # A 2D matrix such that the lengt of sides have int squares
    Ny, Nz =  A.shape
    Ny_s = int(np.sqrt(Ny))
    Nz_s = int(np.sqrt(Nz))
    B = np.zeros((Ny_s, Nz_s))
    for i in range(Ny_s):
        for j in range(Nz_s):
            B[i,j] = np.sum(A[Ny_s*i:Ny_s*(i+1), Nz_s*j:Nz_s*(j+1)])
    return B

def Sympl_DNeg(p, q, Cst, h, Par):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row,
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, M: scalar, rho: scalar, output: list of coordinates in
    # configuration space containing 2D matrix with value for each ray
    M, rho, a = Par
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b, B_2 = Cst
    
    l_abs = np.abs(l)
    x = 2*(l_abs - a)/(np.pi*M)
    r = rho + M*(x*np.arctan(x) - 0.5*np.log(1 + x**2))
    dr = 2/np.pi*np.arctan(x)*np.sign(l)
    d2r = (4*M)/(4*a**2 + M**2*np.pi**2 + 4*l**2 - 8*a*l_abs)
    
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
    return ([
        p_l + p_l_h*h + p_l_h2*h_2,
        p_phi,
        p_th + p_th_h*h + p_th_h2*h_2
        ],
        [
        l +l_h*h + l_h2*h_2,
        phi + phi_h*h + phi_h2*h_2,
        theta + theta_h*h + theta_h2*h_2
        ],
        [H, b_C, B2_C])
