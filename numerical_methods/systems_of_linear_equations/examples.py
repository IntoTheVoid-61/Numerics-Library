from sle_solvers import *


# Coefficients matrix
A = np.array([
    [10, -1,  2,  0],
    [-2, 11, -1,  3],
    [ 2, -1, 10, -1],
    [ 0,  3, -1,  8]
], dtype=float)

# RHS vector
b = np.array([6, 25, -11, 15], dtype=float)


#print(f"Jacobi method solutions: {jacobi(A,b)}" )
#print(f"Gauss-Seidel method solutions: {gauss_seidel(A,b)}" )
#print(f"SOR method solutions: {sor(A,b,0.5)}" )