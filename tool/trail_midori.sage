load("lib/Problem.sage")

import ctypes
import autograd
import autograd.numpy as np

from sage.crypto.sboxes import Midori_Sb0 as S

u = matrix(vector([0, 1, 0, -1, 0, 1, 0, -1, 0, -1, 0, 1, 0, -1, 0, -3]) / 4).T
v = matrix(vector([0, 0, 0, -1, 0, 0, 0, -1, 0,  0, 0, 1, 0,  0, 0, -1]) / 2).T

C  = S.linear_approximation_table().T / 8

ctypes.CDLL('/usr/local/lib/libadept.so', mode = ctypes.RTLD_GLOBAL) 
layout = [
    [None, None, None, None, 
     0,    1,    2,    3],
    [0,    1,    2,    3,
     None, None, None, None]
]
weights = [1, 1, 1, 1]

I  = matrix.identity(16)[:, 1:]

E_u  = Ellipsoid(u, constant = True)
E_v  = Ellipsoid(v, constant = True)
E_Cu = Ellipsoid(C*u, constant = True)
E_Cv = Ellipsoid(C*v, constant = True)

E_I  = Ellipsoid(I)
E_C  = Ellipsoid(C * I)
ellipsoids = [
    [E_Cu, E_Cv, E_Cv, E_Cv,
     E_I, E_I, E_I, E_I],
    [E_C, E_C, E_C, E_C,
     E_u, E_v, E_v, E_v]
]
problem = GenericProblem("./lib/cost.so", layout, weights, ellipsoids, "Midori")

solver = Solver()

print("Note: occasionally converges to local minimum (restart if necessary).")
for _ in range(1):
    x = solver.solve(problem, lambda x : True)
    print(x)
    print("log2(Correlation): ", problem.cost(x))

print((C * v).T)
print(v.T)
