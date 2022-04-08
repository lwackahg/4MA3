#Lukas Gabrys - Inverse Iteration with Rayleigh - Winter 2022 - 4MA3
import numpy as np
import sys
# Reading order of matrix
#n = int(input('How many rows in the matrix: '))
n = 3
# Making numpy array of n x n size and initializing to zero for storing matrix
#a = np.zeros((n, n))
a = np.array([[1, -1, 0], [0, -4, 2], [0, 0, -2]])
#a = np.array([[3, 1], [1, 3]])
# Making numpy array n x 1 size and initializing to zero for storing initial guess vector
x = np.random.rand(n)
# Tolerable error
tolerable_error = 0.1
# Reading maximum number of steps
#max_iteration = int(input('Enter maximum number of steps: '))
max_iteration = 25
# Power Method Implementation
ray_old = 100.0
running = True
iteration = 1
eye = np.eye(n)
print('\nChosen Matrix is as follows: ')
for line in a:
    print(' '.join(map(str, line)))
print('\nRandom X Vector: ')
print (x,'.T')
raytop = np.dot(np.transpose(x), np.dot(a, x))
raybot = np.dot(np.transpose(x), x)
ray = raytop / raybot

#ray = np.dot(np.transpose(x),np.dot(a,x))
while running:
    oldx = x
    a1 = np.column_stack((a-ray*eye, oldx))
    for k in range(n):
        pvt = [abs(a1[i][k]) for i in range(k, n)] # Takes absolute values of column 1
        i_max = pvt.index(
            max(pvt)) + k # Takes the largest absolute value from those values and saves what row it was in
        # Check for error case of singular matrix
        assert a1[i_max][k] != 0, "No solution - Singular Matrix!"
        # Swap lower value row with row that contains the larger absolute value
        a1[[k, i_max]] = a1[[i_max, k]]
        # Gauss Elimination
        for i in range(k + 1, n): # Multipliers for current column
            m = a1[i][k] / a1[k][k]
            for j in range(n + 1): # Applying transformation to sub matrix
                a1[i][j] -= m * a1[k][j]
            a1[i][k] = 0
# Back Substitution
    x[n - 1] = a1[n - 1][n] / a1[n - 1][n - 1]
    for i in range(n - 2, -1, -1): # Works backwards through remaining upper triangular system
        x[i] = a1[i][n]
        for j in range(i + 1, n):
            x[i] = x[i] - a1[i][j] * x[j]
        x[i] = x[i] / a1[i][i] # Solving solutions for unknowns of x
# Finding new Eigen value and Eigen vector
    lam_err = np.linalg.norm(x)
    lam_new = np.linalg.norm(x, ord=np.inf)
    x = x / lam_new
    # Displaying Eigen value and Eigen Vector
    print('\nIteration %d' % (iteration))
    print('----------')
    print('Ray: %0.3f' % ray)
    print('Eigen Vector ')
    for i in range(n):
        print('%0.3f ' % x[i], end=" ")
    print('\n')
    raytop = np.dot(np.transpose(x), np.dot(a, x))
    raybot = np.dot(np.transpose(x), x)
    ray = raytop / raybot

    # Checking maximum iteration)
    iteration = iteration + 1
    if iteration > max_iteration:
        print('Not convergent in given maximum iteration!')
        break
    # Calculating error
    error = abs(ray - ray_old)
    ray_old = ray
    running = error > tolerable_error