import itertools
from sage.crypto.boolean_function import BooleanFunction
from sage.crypto.sboxes import Midori_Sb0 as S

def getBaseCase():
    return [
        (1, matrix([[1, 0], [0, 1]])),
        (1, matrix([[0, 1], [1, 0]])),
        (1, matrix([[1, 0], [0, -1]])),
        (-1, matrix([[0, 1], [-1, 0]]))
    ]

def expandTerm(sign, term):
    return [
        (sign,  block_matrix([[term, 0], [0, term]])),
        (sign,  block_matrix([[0, term], [term, 0]])),
        (sign,  block_matrix([[term, 0], [0, -term]])),
        (-sign, block_matrix([[0, term], [-term, 0]]))
    ]

def expand(terms):
    result = []
    for sign, term in terms:
        result.extend(expandTerm(sign, term))
    return result

MC = expand(expand(expand(getBaseCase())))
C = S.linear_approximation_table().T / 8
R.<x0, x1, x2, x3> = BooleanPolynomialRing()
v1 = vector(BooleanFunction(R(
    x3*x2*x1 + x3*x1 + x3 + x2 + x1 + x0
)).walsh_hadamard_transform()) / 16
v2 = vector(BooleanFunction(R(
    x3*x2 + x2 + x1 + x0
)).walsh_hadamard_transform()) / 16

var("k0, k1, k2, k3")
var("l0, l1, l2, l3")
CK = matrix([[1, 0], [0, k0]]).tensor_product(matrix([[1, 0], [0, k1]])).tensor_product(matrix([[1, 0], [0, k2]])).tensor_product(matrix([[1, 0], [0, k3]]), subdivide = False)
# Note: l0 = l1 = 1 because K2 must be completely weak if we want to get a nonzero correlation
CL = CK.subs(k0 = l0, k1 = l1, k2 = l2, k3 = l3).subs(l0 = 1, l1 = 1)
print("Initial: ", v1, v2, v1*v2)
print("Transf.: ", C*v1, C*v2)

def applyMC(v1, v2):
    """ Apply MixColumn to v1 \otimes v2 \otimes v2 \otimes v2. """
    terms = dict()
    for s, A in MC:
        w1 = s*A*v1 / 16
        w2_p = A*v2
        w2_n = -w2_p
        w2_p.set_immutable()
        w2_n.set_immutable()

        if w2_p in terms:
            terms[w2_p] += w1
        elif w2_n in terms:
            terms[w2_n] -= w1
        else:
            terms[w2_p] = w1
    return [(y, x) for (x, y) in terms.items() if y != 0]

# Forward  direction: AddKey, S, MC
terms_fwd = applyMC(C*CK*v1, v2)
print("Forward direction:  rank =", len(terms_fwd), "[prefactor k3 omitted]")
for (w1, w2) in terms_fwd:
    # Note: k3 factors out anyway
    print(w1 / w1[3], w2, w1[3].collect_common_factors().subs(k3 = 1))
print
# Backward direction: MC, S, AddKey
terms_bkw = applyMC(v1, v2)
print("Backward direction: rank =", len(terms_bkw), "[prefactor k2*k3 omitted]")
for (w1, w2) in terms_bkw:
    # Note: k2 and k3 factor out anyway
    t = (CK*C*w1).subs(k2 = 1, k3 = 1)
    print(t / t[3], (CK*C*w2).subs(k0=1,k1=1), t[3])
print

output = True

print("Forward + backward:")
total = 0
total_sum = 0
for i, (u1, u2) in enumerate(terms_fwd):
    # (First factor of) correlation due to first part
    # Note: k3 factors out anyway
    c_11 = u1[3].collect_common_factors().subs(k3 = 1)

    group_sum = 0
    print("i =", i)
    print(u2)
    for j, (w1, w2) in enumerate(terms_bkw):
        # (Second factor of) correlation due to first part
        # Note: l2 and l3 factor out anyway
        # Multiply by 4 here to compensate for norm of first factor in
        #   tensor product
        c_12 = 4 * (CL*C*w1).subs(l2 = 1, l3 = 1)[3]
        
        # Correlation due to second part; we have three identical copies of
        #  this with different keys (but this only affects the sign)
        c_2 = ((CL*C*w2) * u2).subs(l0 = 1, l1 = 1)
        if c_2 == 0:
            continue
        print(j,end='')
        print("  ", c_12,end='')
        print("* (", c_11, ")", end='')# k3 factors out anyway
        print("  ", c_2)
        # There are essentially only 4 possibilities
        group_sum += (c_12 * c_2 ** 3).substitute(l2^3 == l2, l3^3 == l3)

    if group_sum != 0:
        total += 1
        print("Sum =", group_sum, "* (", c_11, ")")
    else:
        print("Sum = 0")
    print()
    total_sum += group_sum * c_11
print()

print("Total number of nonzero terms: ", total)
print("Expression for the correlation: ", total_sum.collect_common_factors())
print("Evaluation of above expression:")
print("(There are essentially only 4 cases, corresponding to the tables in Beierle et.al.)")
# Output all possible values of the correlation:
for i, xs in enumerate(itertools.product([1, -1], repeat = 2)):
    print("Table", i + 1)
    expr = total_sum.subs(l2 = xs[0], l3 = xs[1])
    print(expr)
    for ys in itertools.product([1, -1], repeat = 3):
        print("{:+.6f}".format(float(expr.subs(k0 = ys[0], k1 = ys[1], k2 = ys[2]))))
    print()
