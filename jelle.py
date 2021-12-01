import WormholeRayTracer as wrmhole
import WormholePics as wrmpics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import InbeddingDiagramDNeg as emb
import os


gar = 'four.png'
sat = 'negfour.png'

wormh = wrmhole.wormhole_with_symmetry(t_end = 100, q0 = [12, np.pi, np.pi/2], Nz=250, Ny=500, Par=[0.43/1.42953, 8.6, 4.3], h=0.001, choice=True)
picture = wrmpics.make_pic_quick(wormh, sat, gar)
np.save('picture', picture)
print('Saving picture')
print('Saving picture')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'TestWormhole.png'), picture)
print('Picture saved')
