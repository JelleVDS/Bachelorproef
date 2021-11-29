import WormholePics as pics
import WormholeRayTracer as raytracer
import os
import cv2
import numpy as np
import time

gar = 'wormhole2.jpg'
sat = 'Saturn.jpg'

wormh = raytracer.wormhole_with_symmetry(t_end = 10000 ,q0 = [100, np.pi, np.pi/2], Nz=1000, Ny=2000, Par=[0.43/1.42953, 8.3, 0.43])
start = time.time()
picture = pics.make_pic_quick(wormh, sat, gar)
np.save('picture', picture)
end = time.time()
tijd = end - start
print(f'Picture made in {tijd} s.')
print('Saving picture')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'Interstellar_met_gridUbuntu.png'), picture)
print('Picture saved')
