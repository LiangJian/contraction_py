from tensors import *
import time

X = Tensor((2, 2, 3, 5), ('a', 'b1', 'c', 'e'))
X.T[0, 0, 0] = 1+1j
Y = Tensor((2, 2, 3), ('a', 'b1', 'd'))
print(X.N, Y.N)

print(X.contract(Y)[0], X.contract(Y)[1].N, X.contract(Y)[1].T.shape)
print(X.contract(Y, result_str_=('a', 'c', 'e', 'd'))[0])


start = time.time()
P = Tensor((24, 24, 24, 64, 4, 3, 4, 3), ('x', 'y', 'z', 't', 'a1', 'c1', 'a2', 'c2'))
Q = Tensor((24, 24, 24, 64, 4, 3, 4, 3), ('x', 'y', 'z', 't', 'a2', 'c2', 'a1', 'c1'))
C = P.contract(Q, conjugate_=True, result_str_=('x', 'y', 'z', 't'))
print(C[0], C[1].N, C[1].T.shape)
end = time.time()
print('%3.1f s used' % (end-start))
