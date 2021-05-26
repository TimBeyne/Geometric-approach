import ctypes, pymanopt, autograd
import autograd.numpy as np
import pymanopt.manifolds
import pymanopt.solvers
import shutil

def orthonormal_basis(V):
    """Compute an orthonormal basismatrix for a given real vector space.
    
    The vector space ``V'' does not need to be defined over V, but it must
     admit conversion through ``V.change_ring(RDF)''.
    The basis matrix is obtained through QR-decomposition.
    """
    W = V.change_ring(RDF)
    Q, _ = W.basis_matrix().T.QR()
    return Q[:, :V.dimension()]

class Ellipsoid:
    def __init__(self, basismatrix, constant = False):
        self.basis = basismatrix
        self.constant = constant
        if not constant:
            self.manifold = pymanopt.manifolds.Sphere(self.dim())

    def dim(self):
        return self.basis.ncols()

    def ambient_dim(self):
        return self.basis.nrows()

def list_to_c_double_array(l):
    return (ctypes.c_double * len(l))(*(float(x) for x in l))

def flatten(v):
    size = sum(len(u) for u in v)
    return (ctypes.c_double * size)(*(x for u in v for x in u))

class StateSpace:

    def __init__(self, layout, ellipsoids):
        """Constructor for ``StateSpace'' objects.
        
        INPUT:
       
        - ``layout'' - list indicating the tensor product structure.
            For example, ``[0, 0, 0, 0]'' corresonds to a state of
            type v^{\otimes 4}.

        - ``ellipsoids'' - list of as many ``Ellipsoid'' objects as
            there are integers in ``layout''. The ellipsoids must be
            compatible in the sense that if layout[i] == layout[j],
            then ellipsoids[i] and ellipsoids[j] must share the same
            dimensions and must be either both constant or both
            variable
        """

        if min(i for i in layout if i is not None) != 0:
            raise Exception("Layout indexing must start from zero.") 

        if len(ellipsoids) != len(layout):
            raise Exception("Mismatching layout and ellipsoid lengths.") 

        # Check correctness of layout
        self.nb_variables = max(i for i in layout if i is not None) + 1
        for i in range(len(layout)):
            if layout[i] is None:
                if not ellipsoids[i].constant:
                    raise Exception("Ellipsoid %d must be constant." % k)
                continue
            for j in range(len(layout)):
                if j is not None and layout[i] == layout[j]:
                    if ellipsoids[i].dim() != ellipsoids[j].dim():
                        raise Exception("Mismatching ellispoids dimensions.")

        self.layout = layout
        self.ellipsoids = ellipsoids

        self.manifold = pymanopt.manifolds.Product([
            e.manifold for e in self.representative_ellipsoids()
        ])

    def representative_ellipsoids(self):
        """Iterator for (nonconstant) representative ellipsoids in order of index."""
        for i in range(self.nb_variables):
            e = self.ellipsoids[self.layout.index(i)]
            yield e

    def layout_no_constants(self):
        """Iterator for the layout indices without counting constants."""

        positions = dict()
        j = 0
        #for i in range(max(self.layout) + 1):
        for (k, i) in enumerate(self.layout):
            if not self.ellipsoids[k].constant:
                if i not in positions:
                    positions[i] = j
                    j += 1
                yield positions[i]
            else:
                yield None

    def embed_fill_constants(self, v):
        """Returns embedding of v in the state space.
           Fills in constants as necessary.
        """
        lst = []
        for (i, j) in enumerate(self.layout):
            if j is None:
                lst.append(np.array([1]))
            else:
                lst.append(v[j])
        return lst

class Problem:

    def __init__(self, statespace):
        self.statespace = statespace

        # Setup pymanopt problem
        self.pymanopt_problem = pymanopt.Problem(
            manifold = self.statespace.manifold,
            cost = self.cost, egrad = self.egrad, ehess = self.ehess,
            verbosity = 2)

    def cost(self, x):
        raise NotImplementedError()

    def egrad(self, x):
        raise NotImplementedError()

    def ehess(self, v):
        print("ERROR: Hessian not implemented.")
        return 0

class ExternalProblem(Problem):
    nb_instances = 0

    """Constructor for ``ExternalProblem'' objects.

        An ``ExternalProblem'' is a basic optimization problem with externally
        defined cost functions. The C API is defined in cost.cpp.
        
        Loads the given DLL containing the external cost and gradient functions.
        Sets up to basismatrices according to the given StateSpace object.
        Creates a corresponding ``pymanopt.Problem'' object.
        
        INPUT:
       
        - ``lib'' - filename of dynamic C library which should contain
            functions ``cost'', ``egrad'', ``register_basismatrix'' and
            ``initalize''.

        - ``statespace'' - an object of type ``StateSpace''.

        - ``linear_layer'' - string passed to the ``inialize'' function.

    """
    def __init__(self, lib, statespace, linear_layer):
        Problem.__init__(self, statespace)
        # Create new instance of the library
        lib_filename = lib + ".%d" % ExternalProblem.nb_instances
        shutil.copy(lib, lib_filename)
        ExternalProblem.nb_instances += 1
        self.lib = ctypes.CDLL(lib_filename, mode = ctypes.RTLD_LOCAL) 
        self.lib.cost.restype = ctypes.c_double

        # Setup prototypes for lib
        double_ptr = ctypes.POINTER(ctypes.c_double)
        self.dblptr_in  = ctypes.c_double * int(sum(f.dim() for f in self.statespace.ellipsoids))
        self.lib.cost.argtypes = [double_ptr]
        self.lib.egrad.argtypes = [double_ptr, self.dblptr_in]
        self.lib.register_basismatrix.argtypes = [ctypes.c_size_t, ctypes.c_size_t, double_ptr]
        self.lib.initialize.argtypes = [ctypes.c_char_p]

        # Setup basismatrices
        for ellipsoid in self.statespace.ellipsoids:
            self.lib.register_basismatrix(
                ellipsoid.dim(), ellipsoid.ambient_dim(),
                list_to_c_double_array(ellipsoid.basis.list())
            )

        self.lib.initialize(ctypes.create_string_buffer(
            bytes(linear_layer, encoding = "ascii")))


    def cost(self, v):
        x = self.statespace.embed_fill_constants(v)
        cost = self.lib.cost(flatten(x))
        return cost

    def extract_factors(self, x):
        factors = []

        j = 0
        for ellipsoid in self.statespace.ellipsoids:
            n = ellipsoid.dim()
            factors.append(np.array(x[j:j+n]))
            j += n
        return factors
    
    def egrad(self, v):
        grad_x = self.dblptr_in()
        x = self.statespace.embed_fill_constants(v)
        self.lib.egrad(flatten(x), grad_x)
        gradient = self.extract_factors(grad_x)
        grad_v = [np.zeros(e.dim()) for e in self.statespace.representative_ellipsoids()]
        for (i, j) in enumerate(self.statespace.layout):
            if j is not None:
                grad_v[j] += gradient[i]
        return grad_v

def inner_product(u, v):
    result = 1
    for i in range(len(v)):
        result *= np.dot(u[i], v[i])
    return result

class BarrierFunction:
    """Keeps track of previously seen states and defines a barrier function,
       which penalizes states which are too close to these.

       This can be used (through ``BarrierSolver'') to enforce inequality constraints.
       """

    def __init__(self):
        self.penalty_vectors = []
        self.penalty_factor  = 0
        self.barrier_grad = autograd.grad(self.barrier)

    def barrier(self, v):
        cost = 0
        for u in self.penalty_vectors:
            #cost -= np.log(1 - abs(inner_product(u, v)))
            abs_in_product = abs(inner_product(u, v))
            if abs_in_product == 1:
                raise Exception("Too close to barrier (low-dimensional space?).")
            cost += 1 / (1 - abs_in_product)^2
        return self.penalty_factor * cost

    def barrier_egrad(self, v):
        if self.penalty_factor != 0:
            return self.barrier_grad(v)
        else:
            return None

class Solver:

    def __init__(self):
        self.solver = pymanopt.solvers.ConjugateGradient(logverbosity = 0)

    def solve(self, problem, condition, x0 = None, max_attempts = 100):
        for i in range(max_attempts):
            x = self.solver.solve(problem.pymanopt_problem, x = x0)
            if condition(x):
                return x
        return None

class BarrierSolver(Solver):

    class BarrierProblemWrapper(Problem):

        def __init__(self, problem, barrier):
            self.problem = problem
            self.barrier = barrier
            Problem.__init__(self, self.problem.statespace)

        def cost(self, x):
            return self.problem.cost(x) + self.barrier.barrier(x)

        def egrad(self, x):
            barrier_gradient = self.barrier.barrier_egrad(x)
            if barrier_gradient is not None:
                return self.problem.egrad(x) + barrier_gradient
            else:
                return self.problem.egrad(x)

    def __init__(self, problem):
        Solver.__init__(self)
        self.invariants = []
        self.barrier = BarrierFunction()
        self.problem = self.BarrierProblemWrapper(problem, self.barrier)
        # Note: we set the penalty vectors by reference
        self.barrier.penalty_vectors = self.invariants

    def solve_barrierfree(self, cost_target, eps):
        self.barrier.penalty_factor = 0
        opt = Solver.solve(
            self, self.problem,
            lambda x : problem.cost(x) < cost_target + eps
        )
        if opt is not None:
            # Invariant, save solution        
            print(opt, self.problem.cost(opt))
            self.invariants.append(opt)
            return True

        return False

    def solve_once(self, correlation_target, max_attempts = 50, eps = 1e-5, print_extra = False):
        log_correlation_target = N(abs(log(correlation_target, 2)))

        # Use barrier method to find new invariants
        # Start from a random guess until
        #   - a new invariant new is found or
        #   - the maximum number of attempts is exceeded
        for _ in range(max_attempts): 
            # First time: no barriers
            if self.invariants == [] and self.solve_barrierfree(log_correlation_target, eps):
                return True

            self.barrier.penalty_factor = 10 # Start with high penalty
            opt = Solver.solve(self, self.problem, lambda _: True, max_attempts = 1)
            if print_extra:
                print(self.barrier.barrier(opt), self.problem.cost(opt))
            while self.barrier.penalty_factor >= 1e-4:
                opt = Solver.solve(self, self.problem, lambda _: True, x0 = opt, max_attempts = 1)
                if print_extra:
                    print(self.problem.cost(opt), self.barrier.barrier(opt))
                self.barrier.penalty_factor /= 1.25 

            # Possibly found something, check
            self.barrier.penalty_factor = 0
            last_cost = self.problem.cost(opt)
            print("Candidate with log2(correlation)", last_cost)
            if last_cost < log_correlation_target + eps:
                print("New solution: ", opt, last_cost)
                self.invariants.append(opt)
                return True

        return False


class GenericProblem(Problem):
    """Represents generic problems involving more than one linear layer. """

    def __init__(self, lib, layout, weights, ellipsoids, linear_layer):
        self.layout = layout
        self.weights = weights
        self.variables = [[x for x in set(l) if x is not None] for l in self.layout]

        self.ellipsoids = ellipsoids

        statespace = StateSpace(
            reduce(operator.add, self.layout), 
            reduce(operator.add, self.ellipsoids)
        )
        Problem.__init__(self, statespace)

        self.ext_problems = []
        for i in range(len(self.layout)):
            statespace = self.local_statespace(i)
            self.ext_problems.append(ExternalProblem(lib, statespace, linear_layer))

    def local_statespace(self, index):
        layout = []
        for i in self.layout[index]:
            if i is None:
                layout.append(None)
            else:
                layout.append(self.variables[index].index(i))

        return StateSpace(layout, self.ellipsoids[index])

    def cost(self, v):
        x = self.statespace.embed_fill_constants(v)
        cost = 0
        j = 0
        for i in range(len(self.layout)):
            layout_len = len(self.layout[i])
            cost += self.weights[i] * self.ext_problems[i].lib.cost(flatten(x[j:j+layout_len]))
            j += layout_len
        return cost

    def egrad(self, v):
        x = self.statespace.embed_fill_constants(v)
        gradient = [np.zeros(e.dim()) for e in self.statespace.representative_ellipsoids()]
        k = 0
        for index in range(len(self.layout)):
            layout_len = len(self.layout[index])
            grad_x = self.ext_problems[index].dblptr_in()
            self.ext_problems[index].lib.egrad(flatten(x[k:k+layout_len]), grad_x)
            temp = self.ext_problems[index].extract_factors(grad_x)
            for (i, j) in enumerate(self.ext_problems[index].statespace.layout):
                if j is not None:
                    gradient[self.variables[index][j]] += self.weights[index] * temp[i]
            k += layout_len
        return gradient
