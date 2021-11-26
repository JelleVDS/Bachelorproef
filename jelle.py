import WormholePics as pics
import WormholeRayTracer as raytracer
import os
import cv2
import numpy as np
import time

gar = 'wormhole.jpg'
sat = 'Saturn.jpg'

wormh = raytracer.wormhole_with_symmetry(tijd = 2000, Nz=1024, Ny=2048, q0 = [96.75, np.pi, np.pi/2], Par=[0.43/1.42953, 8.6, 0.43])
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
