# Geometric-approach
Finding rank-one trails using Riemannian optimization.

To run the tool, the following steps are required (on a Linux or Unix-type system):
- Install [Pymanopt](https://pymanopt.org/) using `sage -pip install pymanopt`.
- Install [Adept](http://www.met.reading.ac.uk/clouds/adept/) for automatic differentiation.
- Compile the file `cost.cpp` into a shared object. Use `g++ -O3 -fPIC -shared cost.cpp -ladept -o cost.so` for gcc.
- Use Sage to execute `find_invariants.sage` or `trail_midori.sage`.
