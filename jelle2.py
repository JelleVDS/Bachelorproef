import WormholeRayTracer as wrmhole
import WormholePics as wrmpics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import InbeddingDiagramDNeg as emb
import os


sat = 'negfourfull16thick.png'
gar = 'fourfull.png'

wormh = wrmhole.wormhole_with_symmetry(t_end = 150, q0 = [6.95, np.pi, np.pi/2], Nz=1024, Ny=2048, Par=[0.014, 1, 0.9], h=10**-10, choice=True)
picture = wrmpics.make_pic_quick(wormh, sat, gar)
print('Saving picture...')
path = os.getcwd()
cv2.imwrite(os.path.join(path, 'testJelle4.png'), picture)
print('Picture saved')
