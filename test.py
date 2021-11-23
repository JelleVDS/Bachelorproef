#Dit is een test file, je kan hier eens wat  uitproberen
print("Hello world!")
#comment comment op een nieuwe branch
print("Hello from the other side")


import numpy as np
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

def numpy_sin(value):
    return np.sin(value)

if __name__ == '__main__': 
    a = np.arange(400*400).reshape(400,400)
    
    start = time.time()
    cores = cpu_count()
    a_P = np.array_split(a, cores, axis=1)
    with Pool(cores) as p:
            P = p.map(numpy_sin, a_P)
    end = time.time()
    print(end - start)
    
    start = time.time()
    b = np.sin(a)
    end = time.time()
    print(end - start)

