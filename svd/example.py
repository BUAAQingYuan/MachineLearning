__author__ = 'PC-LiNing'

from svd import svd
import numpy as np

matrix = np.asarray([1.0,1.0,0.5,1.0,1.0,0.25,0.5,0.25,2.0]).reshape(3,3)
print(matrix)
print(matrix.shape)
singularValues, us, vs = svd(matrix)
print(singularValues)
print(us)
print(vs)
print('#######')
# sum
result = np.zeros(shape=(3,3))
for i in range(3):
    singularValue = singularValues[i]
    u = us[i]
    v = vs[i]
    result += singularValue * np.outer(u, v)
print(result)