from tqdm.auto import tqdm
import cv2
import numpy as np
from math import floor


def read_pics(saturn, gargantua):
    
    print('Reading in pictures...')
    img_saturn    = cv2.imread(saturn)
    img_gargantua = cv2.imread(gargantua)
    
    return img_saturn, img_gargantua
    

def photo_to_sphere(photo):
    """
    Give the pixels of the pictures a spherical coordinate
    Input:  - photo: de pixels van de photo in sferische coordinaten
    Output: - dict: een dictionary met als sleutel (theta, phi) en als waarde
              de RGB-value van de bijbehorende pixel
    """

    dict = {}
    vertical   = len(photo)     #1024
    horizontal = len(photo[0])  #2048
    for row in range(0, vertical):
        for column in range(0, horizontal):
            theta = (np.pi/vertical) * row #- np.pi
            phi   = (2*np.pi/horizontal) * column
            while phi>2*np.pi:
                phi = phi - 2*np.pi
            while phi<0:
                phi = phi + 2*np.pi
            coordinate = (theta, phi) #Tuple with angles that will be used as key
            pixel      = np.array([photo[row][column]]) #RGB-values
            dict[coordinate] = pixel

    return dict


def determine_theta(Nz, theta):
    """
    Determines the location of the ray in the picture.
    Input:  - Nz: height of the picture in pixels
            - theta: the theta value of the ray
    Output: - i: row of the ray
    """
    i = int(floor(Nz * theta / np.pi))
    return i


def determine_phi(Ny, phi):
    """
    Determines the location of the ray in the picture.
    Input:  - Ny: height of the picture in pixels
            - phi: the phi value of the ray
    Output: - j: column of the ray
    """
    j = int(floor(Ny * phi / (2 * np.pi)))
    return j


def make_picture(photo, gargantua, saturn):
    Nz = len(photo)
    Ny = len(photo[0])
    
    pic = np.empty([Nz, Ny, 3])
    for rij in tqdm(range(0, Nz)):
        for kolom in range(0, Ny):
            element = photo[rij][kolom]
            l, phi, theta = element
            loctheta = determine_theta(Nz, theta)
            locphi = determine_phi(Ny, phi)
            if l < 0:
                pic[rij][kolom] = gargantua[loctheta][locphi]
            else:
                pic[rij][kolom] = saturn[loctheta][locphi]
    return pic


def make_pic_quick(pic, sat, gar):
    
    img_saturn, img_gargantua = read_pics(sat, gar)
    print('Pictures ready!')
    print('Making wormhole...')
    picture = make_picture(pic, img_saturn, img_gargantua)
    print('Wormhole ready!')
    
    return picture
    

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

    # subdivides theta and phi
    Par_phi = np.linspace(0, 2*np.pi, N_a-1)
    Par_th = np.linspace(0, np.pi, N_a-1)

    # Defines point on polar grid
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

    # subdivides theta and phi
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
        #ax[k].set_title(Label[k] + ",  Donker pixels binnenkant scherm, lichte pixels buitenkant")
        ax[k].set_xlabel("number of timesteps taken")
        ax[k].set_ylabel(Label[k])
        ax[k].set_title(Label[k]+" summed over a subdivision of rays")
    plt.tight_layout()
    plt.savefig(os.path.join(path, name), dpi=150)
    #plt.show()


def ray_spread(Nz, Ny):
    # input: Ny: amount of horizontal arrays, Nz: amount of vertical arrays
    # output: cl: color based on deviation of the norm of a ray compared to direction obeserver is facing
            #ind ind of original ray mapped to colormap
    S_c = wrmhole.screen_cart(Nz, Ny, 1, 1)
    S_cT = np.transpose(S_c, (2,0,1))
    n = np.linalg.norm(S_cT, axis=0)
    n_u, ind = np. unique(n, return_inverse=True)
    N = n_u.size
    cl = plt.cm.viridis(np.arange(N)/N)

    return cl, ind


def gdsc(Motion, Par, name, path, geo_label = None, select = None, reduce = False):
    # input: Motion: 5D matrix, the elements being [p, q] with p, q as defined earlier
    #       Par: parameters wormhole
    #       Name: picture/filename
    #       Path: directory
    #       select: Give a list of 2D indices to plot only specific geodesiscs
    #       geo_label: if you're just plotting a list of geodesics (thus its elements in order time, [p,q], coordinate),
    #                   then give here a list of strings that which will be the label of your geodesics, corresponding to the order of your geodesics.
    #       reduce: if true sample geodescics uniformly
    M, rho, a = Par

    if np.any(select == None):
        if np.any(reduce == False):
            Sample = np.transpose(Motion, (1,2,3,0))
        else:
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
    r = wrmhole.dneg_r(l, M, rho, a)
    X, Y = r*np.cos(phi), r*np.sin(phi)
    Z = Dia.imb_f_int(l, Par)

    if np.any(reduce == False):
        for k in range(len(Sample[0,0,0])):
            if np.any(select == None):
                geo_label = geo_label[k]
            else:
                geo_label = str(select[k])
            ax.plot(X[:,k], Y[:,k], Z[:,k], label= geo_label)
        ax.scatter(X[0,0] , Y[0,0], Z[0,0], label='camera', c = 'r')
        ax.set_title("Geodesics corresponding to labeled pixel")
        ax.legend()
    else:
        for i in range(Nz_s):
            for j in range(Ny_s):
                ij = i + Nz_s*j
                cl_i =cl[ind[ij]]
                ax.plot(X[:,i,j], Y[:,i,j], Z[:,i,j], color = cl_i, alpha=0.5)
        ax.scatter(X[0,0,0] , Y[0,0,0], Z[0,0,0], label='camera', c = 'r')
        ax.set_title("Geodesics")
        ax.legend()

    # adds surface

    S_l = np.linspace(np.max(l), np.min(l), len(l)+1)
    S_phi = np.linspace(0, 2*np.pi, len(l))
    S_L, S_PHI = np.meshgrid(wrmhole.dneg_r(S_l, M, rho, a), S_phi) # radius is r(l)

    # tile want symmetrisch voor rotaties, onafhankelijk van phi
    # Integraal voor Z richting zoals gedefinieerd in de paper
    S_Z = np.tile(Dia.imb_f_int(S_l, Par), (len(l), 1)) #z(l)

    S_X, S_Y = S_L*np.cos(S_PHI), S_L*np.sin(S_PHI)
    ax.plot_surface(S_X, S_Y, S_Z, cmap=plt.cm.YlGnBu_r, alpha=0.5)
    plt.savefig(os.path.join(path, name), dpi=150)
