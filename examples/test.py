from tensors import *
import numpy
import time


# X = Tensor((2, 2, 3, 5), ('a', 'b1', 'c', 'e'))
# X.T[0, 0, 0] = 1+1j
# Y = Tensor((2, 2, 3), ('a', 'b1', 'd'))
# print(X.N, Y.N)

# print(X.contract(Y)[0], X.contract(Y)[1].N, X.contract(Y)[1].T.shape)
# print(X.contract(Y, result_str_=('a', 'c', 'e', 'd'))[0])


dims = (24, 24, 24, 64, 4, 3, 4, 3)

P = Tensor(dims, ('x', 'y', 'z', 't', 'a1', 'c1', 'a2', 'c2'),np.random.random(dims))
Q = Tensor(dims, ('x', 'y', 'z', 't', 'a2', 'c2', 'a3', 'c1'),np.random.random(dims))
start = time.time()
C = P.contract(Q, conjugate_=True, result_str_=('x', 'y', 'z', 't'))
print(C[0], C[1].N, C[1].T.shape)
end = time.time()
print('%f s used' % (end-start))

