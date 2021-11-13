import WormholeRayTracer as wrmhole
import WormholePics as wrmpics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import InbeddingDiagramDNeg as emb
import os



img_saturn    = cv2.imread('four400.png')
img_gargantua = cv2.imread('negfour400.png')

saturn      = wrmpics.photo_to_sphere(img_saturn)
print('Saturn image loaded.')
gargantua   = wrmpics.photo_to_sphere(img_gargantua)
print('Gargantua image loaded.')

raytracer = wrmhole.wormhole_with_symmetry(tijd=100, initialcond = [12, np.pi, np.pi/2], Nz=400, Ny=400, Par=[0.43/1.42953, 8.6, 43])
# print(raytracer.shape)
print('Ray tracer solution loaded.')
print('Starting image placing process...')
pic = wrmpics.decide_universe(raytracer, saturn, gargantua)
print(pic)
# print(pic)
print('Image placing completed.')
print('Saving picture')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'FOURCOLORS_t=100__L=12__W=0.43__rho=8.6__a=43.png'), pic)
print('Picture saved')
