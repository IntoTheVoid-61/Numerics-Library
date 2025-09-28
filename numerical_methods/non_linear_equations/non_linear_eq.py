import numpy as np
import sympy as sp
"""
The module provides solvers for both single equations and systems of non-linear equations.

It provides solvers of different types such as:
    - Interval methods: Guarantee convergence but can be slow
    - Open methods: Require a good initial guess, faster but dont guarantee convergence
    - Hybrid: Combine positive aspects from Interval and Open methods
    - Systems of non-linear equations


Author: Ziga Breznikar
Date: 28.09.2025

"""


def bisection(func,var,val,stop_crit=10e-5,max_iter=1000):
    """
    Solves f(x) = 0 using bisection method.
    These require an interval [a,b] where f(a) and f(b) have opposite signs.
    They guarantee convergence, but can be slow.

    Parameters
    ----------

    func : sympy expression
        Function to solve.

    var : sympy symbol
        Variable of the function.

    val : tuple or list
        Interval [a, b] with f(a) and f(b) of opposite signs.

    stop_crit : float
        Convergence tolerance (default 1e-5).
    
    max_iter : int
        Maximum number of iterations (default 1000).

    Returns
    -------
    float
        Approximate root.

    """

    func = sp.lambdify(var,func,"numpy")
    a, b = val[0], val[1]
    f_a, f_b = func(a), func(b)

    if f_a * f_b > 0:
        raise ValueError("Function values at interval endpoints must be of opposite signs")
    
    
    for _ in range(max_iter):
        c = (a + b) / 2
        f_c = func(c)

        if abs(f_c) <= stop_crit:
            return c
        
        if f_b * f_c > 0:
            b, f_b = c, f_c
        else:
            a, f_a = c, f_c
        
    print("Did converge returning closest approximation!")
    return c


def newton_raphson(func,var,val,stop_crit=10e-5,max_iter=1000):
    """
    Solves f(x) = 0 using Newton-Raphson method.
    These dont require an initial interval, just a starting guess. They can converge much faster, but may diverge if the guess is poor.
    Can diverge if derivative is small or initial guess is poor.

    Parameters
    ----------

    func : sympy expression
        Function to solve.

    var : sympy symbol
        Variable of the function.

    val : float or int
        Initial guess.

    stop_crit : float
        Convergence tolerance (default 1e-5).
    
    max_iter : int
        Maximum number of iterations (default 1000).

    Returns
    -------
    float
        Approximate root.

    """
    func_der = sp.diff(func,var)
    func = sp.lambdify(var,func,"numpy")
    func_der = sp.lambdify(var,func_der,"numpy")

    for _ in range(max_iter):
        val = val - (func(val) / func_der(val))
        if abs(func(val)) <= stop_crit:
            return val
        
    print("Did converge returning None!")
    return None


def secant(func,var,val,stop_crit=10e-5,max_iter=1000):
    """
    Solves f(x) = 0 using Secant method. 
    Like Newton-Raphson, but avoids computing derivative.
    Convergence: super-linear (~1.6 order), if the initial guesses are close to the root.

    Parameters
    ----------

    func : sympy expression
        Function to solve.

    var : sympy symbol
        Variable of the function.

    val : tuple or list
        Tuple or list of initial guesses (x_0,x_1)

    stop_crit : float
        Convergence tolerance (default 1e-5).
    
    max_iter : int
        Maximum number of iterations (default 1000).

    Returns
    -------
    float
        Approximate root.

    """

    func = sp.lambdify(var,func,"numpy")
    val = list(val) # Tuples are immutable (workaround)


    for _ in range(max_iter):
        temp = val[1]
        val[1] = val[1] - func(val[1]) * (val[1] - val[0]) / (func(val[1]) - func(val[0]))
        val[0] = temp
        if abs(func(val[1])) <= stop_crit:
            return val[1]
        
    print("Did converge returning None!")
    return None


def dekkers_method(func,var,val,stop_crit=10e-5,max_iter=1000):
    """
    Solve f(x) = 0 using Dekker's method (hybrid bisection + secant).

    Parameters
    ----------

    func : sympy expression
        Function to solve.

    var : sympy symbol
        Variable of the function.

    val : tuple or list
        Interval [a, b] with f(a) and f(b) of opposite signs.

    stop_crit : float
        Convergence tolerance (default 1e-5).

    max_iter : int
        Maximum iterations (default 1000).

    Returns
    -------
    float or None
        Approximate root or None if not converged.    

    """
    func = sp.lambdify(var,func,"numpy")
    a, b = val
    f_a, f_b = func(a), func(b)

    if f_a * f_b > 0:
        raise ValueError("Function values at interval endpoints must be of opposite signs")
    
    b_prev, f_b_prev = a, f_a

    for _ in range(max_iter):

        m = (b + a) / 2
        # Compute s
        if f_b != f_b_prev:
            s = b - (b - b_prev) * f_b / (f_b - f_b_prev)
        else:
            s = m
        
        # Define new b
        b_prev = b
        if min(b,m) < s < max(b,m):
            new_b = s       
        else:
            new_b = m

        f_new_b = func(new_b)
        b_prev, f_b_prev = b, f_b
        b, f_b = new_b, f_new_b

        if f_a * f_b < 0:
            pass 
        else:
            a, f_a = b_prev, f_b_prev

        if abs(func(a)) < abs(func(b)):
            a, b = b, a
            f_a, f_b = f_b, f_a
        
        if abs(f_b) <= stop_crit or abs(b - a) < stop_crit:
            return b
    
    print("Did not converge, returning best approximation")
    return b
        

def newton_raphson_multi(func,var,val,stop_crit=10e-5,max_iter=1000):
    """
    Solve a system of nonlinear equations F(x) = 0 using the
    Newton-Raphson method (multi-variable version).

    Parameters
    ----------
    func : sympy.Matrix
        Vector of nonlinear equations [f1, f2, ..., fn].
    var : list of sympy.Symbol
        Variables [x1, x2, ..., xn].
    val : numpy.ndarray
        Initial guess for the variables.
    stop_crit : float
        Convergence tolerance (default 1e-5).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    numpy.ndarray or None
        Approximate solution vector, or None if not converged.    
    
    """
    jacobian = func.jacobian(var)
    

    func = sp.lambdify(var,func,"numpy")
    jacobian = sp.lambdify(var,jacobian,"numpy")

    for _ in range(max_iter):
        val = val - np.linalg.inv(
            np.array(jacobian(*val), dtype=float)
            ) @ np.array(func(*val), dtype=float).flatten()
        if np.max(func(*val)) <= stop_crit:
            return val
    
    print("Did converge returning None!")
    return None
    
    

























