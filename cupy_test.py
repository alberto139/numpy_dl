import numpy as np
import cupy as cp
import time


n = 700
### Numpy and CPU
s = time.time()
x_cpu = np.ones((n,n,n))
e = time.time()
numpy_time = (e - s)


### CuPy and GPU
s = time.time()
x_gpu = cp.ones((n,n,n))
#cp.cuda.Stream.null.synchronize()
e = time.time()
cupy_time = (e - s)

print(numpy_time/cupy_time)

### Numpy and CPU
s = time.time()
x_cpu *= 5
e = time.time()
numpy_time = (e - s)
### CuPy and GPU
s = time.time()
x_gpu *= 5
#cp.cuda.Stream.null.synchronize()
e = time.time()
cupy_time = (e - s)

print(numpy_time/cupy_time)