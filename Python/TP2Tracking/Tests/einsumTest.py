import numpy as np
a = np.array([10,12])
b = np.array([5,10])
mat_z = a - b
answer = np.sqrt(np.einsum('i,i->', mat_z, mat_z))
print(answer)
print(mat_z)


#  https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
# https://semantive.com/high-performance-computation-in-python-numpy-2/