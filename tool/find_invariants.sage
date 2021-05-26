load("lib/Problem.sage")

import ctypes
import autograd
import autograd.numpy as np

# Note: can be used with any other 4-bit S-box.
from sage.crypto.sboxes import Midori_Sb0 as S

# Unconstrained problem: 
basismatrix = matrix.identity(16)[:, 1:]
# Joint invariant: 
# +1 eigenspcae
#E = (S.linear_approximation_table().T / 8 - matrix.identity(16)).right_kernel()
#basismatrix = orthonormal_basis(E)
# -1 eigenspcae
#E = (S.linear_approximation_table().T / 8 + matrix.identity(16)).right_kernel() 
#basismatrix = orthonormal_basis(E)[:, 1:]
ellipsoids = [Ellipsoid(basismatrix) for _ in range(8)]
statespace = StateSpace([0, 1, 2, 3, 0, 1, 2, 3], ellipsoids)


# Load library
ctypes.CDLL('/usr/local/lib/libadept.so', mode = ctypes.RTLD_GLOBAL) 
# Implemented linear layers: Midori, Qarma, Skinny, Saturnin, GIFT, LED
problem = ExternalProblem("./lib/cost.so", statespace, "Midori")

# Note: modify to find more than one invariant.
# This relies on barrier methods and can be significantly optimized by 
#   exploiting (e.g.) orthogonality.
nb_invariants = 1
solver = BarrierSolver(problem)

if input("Restart from checkpoint? [y/*]") != "y":
    previous_invariants = []
else:
    try:
        print("Loading checkpoint.")
        previous_invariants = load("invariants_checkpoint")
    except FileNotFoundError:
        print("No checkpoint file not found.")
        exit()

solver.invariants.extend(previous_invariants)
while solver.solve_once(1):
    print("New invariant found, total: {}.".format(len(solver.invariants)))
    print()
    previous_invariants = copy(solver.invariants)
    save(previous_invariants, "invariants_checkpoint")

    if len(solver.invariants) >= nb_invariants:
        break



#print(solver.invariants)
