import numpy as np
from scipy.linalg import solve, eigh
from scipy.optimize import approx_fprime, rosen

def newton_method(f, x_0, N, damping_factor = 0.5, eps=1e-6):
    x_values = [x_0]
    f_values = [f(x_0)]

    for i in range(N):
        gradient = approx_fprime(x_values[-1], f)
        hessian = approx_hessian(x_values[-1], f)

        if not is_positive_definite(hessian):
            hessian = hessian + damping_factor * np.eye(len(x_0))

        d = solve(hessian, -gradient)
        
        x_values.append(x_values[-1] + d)
        f_values.append(f(x_values[-1]))

        if np.linalg.norm(d) < eps:
            break

    print('Newton\'s method performed ' + str(i+1) + ' iterations')
    return x_values, f_values

def is_positive_definite(matrix, tol = 1e-04):
    
    # Compute the eigenvalues
    eigenvalues = eigh(matrix, eigvals_only = True)
    return np.all(eigenvalues > tol)


def approx_hessian(x, f):
    n = len(x)
    hessian = np.zeros((n, n))

    for i in range(n):
        def grad_i(y):
            return approx_fprime(y, f)[i]
        hess_i = approx_fprime(x, grad_i,epsilon=1e-6)
        
        for j in range(n):
            if i <= j:
                hessian[i, j] = hess_i[j]

                hessian[j, i] = hessian[i, j]
    return hessian


def gradient_descent(f, x_0, alpha_0=0.05, apx_LS=False, N=50, eps = 1e-4):
    x_values = [x_0]
    f_values = [f(x_0)]

    for i in range(N):
        d = -approx_fprime(x_values[-1], f)

        if apx_LS:
            alpha = apx_line_search(f, x_values[-1], d, alpha_0=alpha_0)
        else:
            alpha = alpha_0
            
        # Update x
        x_new = x_values[-1] + alpha*d
        x_values.append(x_new)
        f_values.append(f(x_new))
        
        # Stopping criterion
        if np.linalg.norm(d)<eps:
            break


    print('Gradient descent method performed ' + str(i+1) + ' iterations')
    return x_values, f_values

def apx_line_search(f, x, d, c = 0.1, t = 0.9, alpha_0 = 1):

    alpha = alpha_0
    f_x = f(x)
    
    def phi(a):
        return f(x+ a*d)
    
    phi_prime = approx_fprime(0, phi)
    
    while  phi(alpha) > f_x +c*alpha*phi_prime:
        alpha *= t
        
    return alpha

# Function to nicely display the results of our algorithm

def display_results(x, values, prec=3):
    np.set_printoptions(precision=prec, suppress=True)
    
    header = f"{'Iteration':<12}{'x Values':<40}{'Function Value':<20}"
    separator = "=" * len(header)

    print(header)
    print(separator)

    for i in range(len(x)):
        x_values = ', '.join(f"{val:.{prec}f}" for val in x[i])
        value_str = f"{values[i]:.{prec}f}"

        print(f"{i + 1:<12}{x_values:<40}{value_str:<40}")

'''x_0 = np.array([1.0, 1.0, 1.0, 1.0])     

     
def f(x):
    return np.sum(x**2)

x, val = gradient_descent(f, x_0, apx_LS=False)

display_results(x, val)

x, val = gradient_descent(f, x_0, apx_LS=True)

display_results(x, val)

def f(x):
    Q = np.diag([100, 10, 1])
    return float(x.T @ Q @ x)

x_0 = np.array([3,3,3])   
            
x, val = newton_method(f, x_0)   

display_results(x, val)  '''