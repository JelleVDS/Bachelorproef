import numpy as np
import cv2
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import RungeKutta as rk
#import InbeddingDiagramDNeg as Dia
import WormholeRayTracer as wrmhole
import WormholePics as WPic
import Symplectic_DNeg as Smpl
import time
import os


if __name__ == '__main__':

    path = os.getcwd()
    Par = [0.43/1.42953, 1, 0] # M, rho, a parameters wormhole
    Integrator = 1 # 1 = scipy integrator, 0 = symplectic intgrator
    #initial position in spherical coord

    if Integrator == 1:
         initial_q = np.array([[17, np.pi, np.pi/2]])
         Grid_dimension = '2D'
         mode = 0
         Motion1, Photo1, CM1 = wrmhole.Simulate_DNeg(Smpl.Sympl_DNeg, Par, 0.02, 1500, initial_q, 20**2, 20**2, Grid_dimension, mode)


    if Integrator == 0:
        Nz = 200
        Ny = 400
        start = time.time()
        sol = wrmhole.simulate_radius(22, Par, [20, np.pi, np.pi/2], Nz, Ny, methode = 'RK45')
        end = time.time()
        print('Tijdsduur = ' + str(end-start))
        momenta, position = sol

         # np.save('raytracer2', position)

        picture = wrmhole.rotate_ray(position, Nz, Ny)
        # print(position)
        print('saving location...')
        np.save('raytracer2', picture)
        print('location saved!')

        print('Saving picture')
        path = os.getcwd()
        cv2.imwrite(os.path.join(path, 'picture2.png'), picture)
        print('Picture saved')


    #print(picture)
    if Integrator == 1 or Integrator == 2: # conditions can be altered once the scipy integrator is able to return geodescics
        if mode ==  0:
             WPic.plot_CM(CM1, ['$H$', '$b$', '$B^{2}$'], "Pictures/CM DNeg Sympl"+str(Par)+" "+str(initial_q)+".png", path)
        #plot_CM(CM2, ['H', 'b', 'B**2'])

        #start = time.time()
        #sol = simulate_raytracer(0.01, 100, [5, 3, 3], Nz = 20**2, Ny = 20**2, methode = 'RK45')
        #end = time.time()
        #print('Tijdsduur = ' + str(end-start))
        #print(sol)
        #np.save('raytracer2', sol)`

        if mode ==  0:
            Geo_label = None #list of strings for labeling geodesics, turn Geo_selto None
            #Geo_Sel = None
            Geo_Sel = [[348, 70], [296, 360], [171, 175], [85, 37], [10, 10]]
            if Geo_Sel == None:
                Geo_txt = ""
            else:
                 Geo_txt = str(Geo_Sel)
            WPic.gdsc(Motion1, Par, "Pictures/geodesics "+Geo_txt+" DNeg Sympl"+str(Par)+" "+str(initial_q)+".png", path, Geo_label, Geo_Sel)

        cv2.imwrite(os.path.join(path, "Pictures/Image "+Grid_dimension+"Gr DNeg Sympl"+str(Par)+" "+str(initial_q)+".png"), 255*Photo1)


