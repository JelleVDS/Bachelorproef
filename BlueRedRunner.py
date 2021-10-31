import WormholeRayTracer as w
import Symplectic_DNeg as Smpl
import os
import numpy as np
import cv2

for teller in range(1, 15):
    Motion1, Photo1, CM1 = w.Simulate_DNeg(Smpl.Sympl_DNeg, 0.01, 1500, np.array([teller, 3, 2]), 20**2, 20**2)
    path = os.getcwd()
    cv2.imwrite(os.path.join(path, f'DNeg Sympl_{teller}.png'), 255*Photo1)
