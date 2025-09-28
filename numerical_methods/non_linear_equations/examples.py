from non_linear_eq import *

#------------------------------Single non-linear equation--------------------------------------#
# Variable
x = sp.Symbol('x')

# Equations
f_1 = x**3 - x - 2
f_2 = sp.cos(x) - x
f_3 = 2.71**-x -x


#print(f"Bisection solution: {bisection(f_2,x,(-2,2))}")
#print(f"Newton-Raphson solution: {newton_raphson(f_2,x,1)}")
#print(f"Secant solution: {secant(f_2,x,[-2,-1.5])}")
#print(f"Dekkers solution: {dekkers_method(f_2,x,(-2,2))}")


#------------------------------System of non-linear equations--------------------------------#

# Variables
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
# Equations
eq1 = sp.sin(x1) + x2**2 + x3 - 3
eq2 = x1 + sp.cos(x2) - x4 - 1
eq3 = x1**2 + x2 + sp.exp(x3) - 6
eq4 = x4 + x3**2 - x2 - 2

# Function vector
func = sp.Matrix([eq1, eq2, eq3, eq4])

# Example initial guess
initial_guess = (1.0, 0.5, -0.5, 1.0)

#print(newton_raphson_multi(func, (x1, x2, x3, x4), initial_guess))