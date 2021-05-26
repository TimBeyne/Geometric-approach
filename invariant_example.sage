import itertools
from sage.crypto.sboxes import SBox

S = SBox((7, 14, 1, 13, 0, 9, 8, 11, 2, 6, 15, 3, 4, 5, 10, 12))

C = S.linear_approximation_table().T / 8

FT = matrix([[1, 1], [1, -1]])
FT = FT.tensor_product(FT, subdivide = False)
FT = FT.tensor_product(FT, subdivide = False)

zeta = I
def f(x):
    if x in [0, 13, 8, 10]:
        return 0
    if x in [7, 5, 2, 15]:
        return 1
    if x in [11, 9, 1, 12]:
        return 2
    if x in [3, 6, 14, 4]:
        return 3

v1 = FT * vector([1] * 16) / 16
v2 = FT * vector([zeta ^ f(x) for x in range(16)]) / 16
v3 = FT * vector([zeta ^ (2 * f(x)) for x in range(16)]) / 16
v4 = FT * vector([zeta ^ (3 * f(x)) for x in range(16)]) / 16

# v1, v2, v3, v4 are eigenvectors of C^S:
assert C * v1 == v1
assert C * v2 == -I * v2
assert C * v3 == -v3
assert C * v4 == I * v4

print("Eigenvector basis:")
print(v1)
print(v2)
print(v3)
print(v4)
print()

print("Real basis:")
w1 = v2 + v4
w2 = (v2 - v4) / I
print(v1)
print((w1 + w2) / 2)
print(v3)
print(w1)
print()
