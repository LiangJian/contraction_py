from tensors import *
import numpy
import time


# X = Tensor((2, 2, 3, 5), ('a', 'b1', 'c', 'e'))
# X.T[0, 0, 0] = 1+1j
# Y = Tensor((2, 2, 3), ('a', 'b1', 'd'))
# print(X.N, Y.N)

# print(X.contract(Y)[0], X.contract(Y)[1].N, X.contract(Y)[1].T.shape)
# print(X.contract(Y, result_str_=('a', 'c', 'e', 'd'))[0])


dims1 = ( 64, 4, 3, 4, 3)
dims1_name = ('x1', 'a1', 'c1', 'a2', 'c2')

dims2 = (24,  4, 3, 4, 3)
dims2_name = ('x2', 'a2', 'c2', 'a1', 'c1')

P = Tensor(dims1,dims1_name,np.random.random(dims1))
Q = Tensor(dims2,dims2_name,np.random.random(dims2))
start = time.time()
C = P.contract(Q, conjugate_=True, result_str_=('x1',  'x2'))
print(C[0], C[1].N, C[1].T.shape)
end = time.time()
print('%f s used' % (end-start))
