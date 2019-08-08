import numpy as np
from multiprocessing import Pool
import time
from functools import partial

def func(a, b):
    return a * b

if __name__ == "__main__":
        
    arg = np.random.randint(0,10, 100)
    start = time.time()
    pool = Pool(processes=4)
    # a = pool.starmap(func, zip([5]*100, arg))
    a = pool.map(partial(func, b=5), arg)
    # a = []
    # for i in arg:
    #     a.append(func(i))
    print(a, "time:{}".format(time.time()-start))