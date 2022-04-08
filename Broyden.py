#Lukas Gabrys - Inverse Iteration - Winter 2022 - 4MA3
# import libraries
import numpy as np

# define broyden
def broyfunc(x, y, f_x, J_eq, tol=1e-4):
    n = 2 # set amount of unknowns
    f = f_x(x, y) # combine initial x's
    s = [10, 10] # set temp x variable
    steps = 0
    Jacbo = J_eq(x, y) # set jacobian initial guess (Identity Matrix)
    while np.linalg.norm(f,2) > tol: # while norm is larger than the set tol

        print("\nB", steps, "\n", np.around(Jacbo, 2))
        print("X", steps, "\n", np.around((x, y), 2))
        print("Fx", steps, "\n", np.around(f,2))

        gausselim(Jacbo,f,s,n) # gauss elim function to solve for unknowns
        x = x + s[0] # adds new x's to old variables
        y = y + s[1]
        nextf = f_x(x, y) # sets new x vector
        epi = nextf - f
        a = (np.outer ((epi - np.dot(Jacbo,s)),s)) / (np.dot(s,s)) # find what to add to Jacobian matrix
        Jacbo = Jacbo + a # new jacobian
        f = nextf # next fx set
        steps += 1

    return steps, x, y

def gausselim(Jacbo,f,s,n):
    a1 = np.column_stack((Jacbo, -1 * f)) # solve negative fx
    for k in range(n):
        # Gauss Elimination
        for i in range(k + 1, n):  # Multipliers for current column
            m = a1[i][k] / a1[k][k]
            for j in range(n + 1):  # Applying transformation to sub matrix
                a1[i][j] -= m * a1[k][j]
            a1[i][k] = 0
    # Back Substitution
    s[n - 1] = a1[n - 1][n] / a1[n - 1][n - 1]
    for i in range(n - 2, -1, -1):  # Works backwards through remaining upper triangular system
        s[i] = a1[i][n]
        for j in range(i + 1, n):
            s[i] = s[i] - a1[i][j] * s[j]
        s[i] = s[i] / a1[i][i]  # Solving solutions for unknowns of x
    return s

def J_eq(x,y):
 return np.eye(2)

# f function
def f_eq(x,y):
 return np.array([x + 2*y - 2, x**2 + 4*y**2 - 4])

tol = 0.0001 # The tolerance
x0 = 1 # initial point x
x1 = 2 # initial point y

n, x, y = broyfunc(x0, x1, f_eq, J_eq, tol)

print("\nFinished in",n,"iterations")
print("Final x and x1: ", round(x,2), round(y,2))
