import numpy as np
"""
A collection of iterative solvers for systems of linear equations (SLEs) of the form Ax = b. 

This module provides educational, lightweight implementations of classical methods such as:
    - Jacobi
    - Gauss-Seidel
    - Successive Over-Relaxation (SOR) 

The solvers are designed for clarity and reusability, making them useful both for learning 
and for applying to small/medium numerical problems. Convergence is guaranteed only under 
specific conditions.

Author: Ziga Breznikar
Date: 28.09.2025


"""

def gauss_seidel(A,b,stop_crit=10e-5,max_iter=100):
    """
    Function is used to solve a system of linear equations.
    A system of linear equations is defined as Ax = b, where A and b are known and x are unknowns.

    It can be applied to any non-zero diagonal matrix A.
    Converges is only guaranteed if A is one the following:
        - Strictly diagonally dominant 
        - Symmetric and positive definite

    Parameters
    -----------

    A: list[float]
        A matrix of coefficients
    
    b: list[float]
        A vector of RHS constants

    
    Returns
    --------

    x_new: np.ndarray
        Vector of solutions to SLE


           
    """
    A = np.array(A,dtype=float)
    b = np.array(b,dtype=float)
    dimension = len(b)

    x_old = np.zeros(dimension,dtype=float)
    x_new = x_old.copy()
    
    for _ in range(max_iter):
        for i in range(dimension):
            sum_1 = np.dot(A[i,:i],x_new[:i])
            sum_2 = np.dot(A[i,i+1:],x_old[i+1:])
            x_new[i] = (b[i] - sum_1 - sum_2) / A[i][i]

        diff = np.linalg.norm(x_new - x_old, ord=np.inf)

        if diff < stop_crit:
            return x_new
        
        x_old = x_new.copy()

    print("Did not converge, returning None")
    return None



def jacobi(A,b,stop_crit=10e-5,max_iter=100):
    """
    
    Function is used to solve a system of linear equations.
    A system of linear equations is defined as Ax = b, where A and b are known and x are unknowns.

    It can be applied to strictly diagonally dominant matrix A.

    Parameters
    -----------

    A: list[float] or list[int]
        A matrix of coefficients

    b: list[float] or list[int]
        A vector of RHS constants

    Returns
    --------

    x_new: np.ndarray
        Vector of solutions to SLE    

    """

    A = np.array(A,dtype=float)
    b = np.array(b,dtype=float)
    dimension = len(b)

    x_old = np.zeros(dimension,dtype=float)
    x_new = x_old.copy()

    for _ in range(max_iter):
        for i in range(dimension):
            sum = np.dot(A[i,:i],x_old[:i])
            sum += np.dot(A[i,i+1:],x_old[i+1:])
            x_new[i] = (b[i] - sum) / A[i][i]

        diff = np.linalg.norm(x_new - x_old, ord=np.inf)

        if diff < stop_crit:
            return x_new
        
        x_old = x_new.copy()

    print("Did not converge, returning None")
    return None    


def sor(A,b,relaxation_fact,stop_crit=10e-5,max_iter=100):
    """
    Successive over-relaxation is a variant of Gauss-Seidel method.
    Method should be used if G-S method is slowly converging.

    The choice of relaxation factor is not necessarily easy and depends,
    upon the properties of coefficient matrix.

    
    Parameters
    -----------
    A: list[float] or list[int]
        A matrix of coefficients

    b: list[float] or list[int]
        A vector of RHS constants

    
    Returns
    --------

    x_new: np.ndarray
        Vector of solutions to SLE         
    """       

    A = np.array(A,dtype=float)
    b = np.array(b,dtype=float)
    dimension = len(b)

    x_old = np.zeros(dimension,dtype=float)
    x_new = x_old.copy()

    for _ in range(max_iter):
        for i in range(dimension):
            sum_1 = np.dot(A[i,:i],x_new[:i])
            sum_2 = np.dot(A[i,i+1:],x_old[i+1:])
            x_new[i] = (1 - relaxation_fact) * x_old[i] + (relaxation_fact / A[i][i]) * (b[i] - sum_1 - sum_2)

        diff = np.linalg.norm(x_new - x_old, ord=np.inf)

        if diff < stop_crit:
            return x_new
            
        x_old = x_new.copy()

    print("Did not converge, returning None")
    return None

