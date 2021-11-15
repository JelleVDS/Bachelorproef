import WormholePics as pics
import WormholeRayTracer as raytracer
import os
import cv2
import numpy as np

sat = 'four_400.png'
gar = 'neg_four_400.png'

wormh = raytracer.wormhole_with_symmetry(tijd = 1000, Nz=100, Ny=100, initialcond = [96.75, np.pi, np.pi/2],Par=[0.43/1.42953, 8.6, 43])
picture = pics.make_wormhole_pic(wormh, sat, gar)
print('Saving picture')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'Pictures/Interstellar_met_grid22.png'), picture)
print('Picture saved')
