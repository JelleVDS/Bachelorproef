import numpy as np

def Sympl_DNeg(p, q, Cst, h, M = 0.43/1.42953, rho = 1):
    # input: p: matrix with coordinates in momentum space on first row,
    # q: matrix with coordinates in configuration space on first row,
    # Cst: list of cst of motion containing the value for each ray in 2D matrix,
    # h: stepsize, M: scalar, rho: scalar, output: list of coordinates in
    # configuration space containing 2D matrix with value for each ray
    p_l, p_phi, p_th = p
    l, phi, theta = q
    b, B_2 = Cst
    
    l_abs = np.abs(l)
    x = 2*l_abs/(np.pi*M)
    x_atan = np.arctan(x)
    r = rho + M*(x*x_atan - 0.5*np.log(1 + x**2))
    dr = 2/np.pi*x_atan
    d2r =  4*M*l/(np.pi**2*M**2*l_abs + 4*l_abs**3)
    
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
    H = 0.5*np.sum((H1 + H2 + H3))

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
        H)
