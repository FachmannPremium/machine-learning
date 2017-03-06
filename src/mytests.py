import numpy as lumpy

m = lumpy.ones((4,4))

# m[1:2] *= 4
m[:,1] += 2

print(m)