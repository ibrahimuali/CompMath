import numpy as np
from scipy.optimize import minimize, approx_fprime, rosen
import matplotlib.pyplot as plt
from scipy.linalg import solve, eigh
import pandas as pd

class NelsonSiegel_SvenssonModel:
    """
    Class to handle the Nelson-Siegel model and its parameter estimation.

    Attributes:
    - beta0, beta1, beta2, tau: float
        These are the parameters of the Nelson-Siegel model.
    - yields: numpy.ndarray
        Array of historical yield data for various securities.
    """

    def __init__(self, yields: np.ndarray, beta0: float, beta1: float, beta2: float, tau: float, beta3 = None, tau2 = None):
        """
        Constructor to instantiate the NelsonSiegelModel class.

        Parameters:
        - beta0, beta1, beta2, tau: float
            Parameters of the Nelson-Siegel model.
        - yields: numpy.ndarray
            Array of historical yield data for various securities.
        """

        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.tau = tau
        self.tau2 = tau2
        self.yields = yields

    def compute_R(self, t: float):
        """
        Computes the R(t) value using the Nelson-Siegel model.

        Parameters:
        - t: float
            The time parameter for which R(t) is to be computed.

        Returns:
        - float:
            The computed R(t) value.
        """
        
        f1 = (1 - np.exp(-t / self.tau)) / (t / self.tau)
        f2 = (1 - np.exp(-t / self.tau)) / (t / self.tau) - np.exp(-t / self.tau)

        if self.tau2 is None:
            R = self.beta0 + self.beta1 * f1 + self.beta2 * f2
        else:
            f3 = (1 - np.exp(-t / self.tau2)) / (t / self.tau2) - np.exp(-t / self.tau2)
            R = self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3
        return R

    def compute_f(self, t: float):
        """
        Computes the f(β0, β1, β2, τ) or f(β0, β1, β2, β3, τ_0, τ_1) value using the Nelson-Siegel model.

        Returns:
        - float:
            The computed f(β0, β1, β2, τ) or f(β0, β1, β2, β3, τ_0, τ_1) value.
        """
        
        residuals = self.yields - np.array([self.compute_R(t) for t in range(1, len(self.yields) + 1)])
        return np.sum(residuals ** 2)

    def newton_method(self, f, x_0, N=100, damping_factor = 0.5, eps=1e-6):
        x_values = [x_0]
        f_values = [f(x_0)]

        for i in range(N):
            gradient = approx_fprime(x_values[-1], f)
            hessian = self.approx_hessian(x_values[-1], f)

            if not self.is_positive_definite(hessian):
                hessian = hessian + damping_factor * np.eye(len(x_0))

            d = solve(hessian, -gradient)
        
            x_values.append(x_values[-1] + d)
            f_values.append(f(x_values[-1]))

            if np.linalg.norm(d) < eps:
                break

        print('Newton\'s method performed ' + str(i+1) + ' iterations')
        return x_values, f_values

    def is_positive_definite(matrix, tol = 1e-04):
        '''
        Checks whether a given square matrix is a positve definite.
        
        Parameter:
        - matrix:
            The input square matrix for which the positive definiteness is checked.
        - tol: float (default: 1e-04)
            A tolerance parameter to account for numerical precision.

        Returns:
        - eigenvalues_all: list
            All eigenvalues greater than tolerance parameter
        '''
        # Compute the eigenvalues
        eigenvalues = eigh(matrix, eigvals_only = True)
        eigenvalues_all = np.all(eigenvalues > tol)
        return eigenvalues_all


    def approx_hessian(x, f):
        '''
        Compute an approximate the Hessian matrix for a givena vector x and a scalar function f
        
        Parameters:
        - x:
            The vector at which to compute the Hessian.
        - f:
            The scalar function for which the Hessian is to be approximated.
            
        Returns:
        - hessian:
            The computed Hessian matrix.
        '''
        
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


    def gradient_descent(self, f, x_0, t, alpha_0=5, apx_LS=False, N=50, eps = 1e-4):
        '''
        Compute the gradient descent algorithm to find the minimum of a given objjective function,
        
        Parameters:
        - f: 
            The objective function to be minimized.
        - x_0: 
            The initial guess or starting point for the optimization.
        - t: float
            A parameter controlling the step size reduction.
        - alpha_0: int (default: 1)
            The initial step size for the gradient descent update.
        - apx_LS: 
            A boolean flag indicating whether to use approximate line search. If True, the line search is performed using the apx_line_search function. If False, a constant step size (alpha_0) is used. The default value is set to False.
        - N: 
            The maximum number of iterations for the gradient descent algorithm. The default value is set to 50.
        - eps: 
            The tolerance or stopping criterion. If the L2 norm of the gradient is less than eps, the optimization process stops. The default value is set to 1e-4.
        '''
        x_values = [x_0]
        f_values = [f(x_0)]

        for i in range(N):
            d = -approx_fprime(x_values[-1], f)

            if apx_LS:
                alpha = self.apx_line_search(f, x_values[-1], d, t, alpha_0=alpha_0)
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

    def apx_line_search(f, x, d, t, c = 0.1, alpha_0 = 1):
        '''
        Parameters:
        - f: 
            The objective function to be minimized.
        - x: list
            The current iterate (representing a point in the optimization space).
        - d: list
            The search direction (representing the direction in which to search for the minimum).
        - c: float (default: 0.1)
            A parameter controlling the sufficient decrease condition.
        - t: float
            A parameter controlling the step size reduction.
        - alpha_0: int (default: 1) 
            The initial guess for the step size.
        
        Returns:
        - alpha: int
            Determined step size.
        '''

        alpha = alpha_0
        f_x = f(x)
    
        def phi(a):
            return f(x+ a*d)
    
        phi_prime = approx_fprime(0, phi)
    
        while  phi(alpha) > f_x +c*alpha*phi_prime:
            alpha *= t
        
        return alpha

    def estimate_parameters(self, method, t, starting_points: list = [0, 0, 0, 1], step_sizes: list = [0.1, 0.1, 0.1, 0.1]):
        """
        Estimates the parameters of the Nelson-Siegel model using the specified optimization method.

        Parameters:
        - method: str (default: 'gradient_descent')
            The optimization method to use. Can be 'gradient_descent' or 'newton'.
        - t: int

        - starting_points: list (default: [0, 0, 0, 1])
            List of starting points for the optimization algorithm.
        - step_sizes: list (default: [0.1, 0.1, 0.1, 0.1])
            List of step sizes for the optimization algorithm.

        Returns:
        - dict:
            A dictionary containing the optimal solutions, optimal values, and number of iterations or running time for each instance and algorithm.
        """

        results = {}
        for i, start_point in enumerate(starting_points):
            if method == 'gradient_descent':
                x, f_values = self.gradient_descent(self.compute_f, start_point, t)
            elif method == 'newton':
                x, f_values = self.newton_method(self.compute_f, start_point)

            results[f'Instance {i+1}'] = {'Optimal Solution': x[-1], 'Optimal Value': f_values[-1], 'Iterations': len(x) - 1}

        return results
    
    def plot_curve(self):
        """
        Plots the curve (t, R(t)) using the computed R(t) values and the historical yield data.
        """

        t_values = range(1, len(self.yields) + 1)
        R_values = [self.compute_R(t) for t in t_values]

        plt.plot(t_values, self.yields, label='Historical Yield Data')
        plt.plot(t_values, R_values, label='Nelson-Siegel Curve')
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Nelson-Siegel Curve vs Historical Yield Data')
        plt.legend()
        plt.show()

# Example usage:

germany_1 = pd.read_csv('Bond/Germany 1-Year Bond Yield Historical Data.csv')
germany_1_df = pd.DataFrame(germany_1)

def clear_df(df):
    '''Drop useless columns and set date as index
    Return: DataFrame'''

    columns_to_drop = ['Open', 'High', 'Low', 'Change %']
    df.drop(columns=columns_to_drop, inplace=True)
    df.set_index('Date', inplace=True)
    df = df[::-1]
    return df

'''germany_1_df_clear = clear_df(germany_1_df)

model = NelsonSiegel_SvenssonModel(germany_1_df_clear['Price'], 0.05, -0.02, -0.07, 1)

# Computing R(t) for a specific time
t = 45
R = model.compute_R(t)
print(f"R({t}) = {R}")

# Computing f(β0, β1, β2, τ)
f = model.compute_f(t)
print(f"f(β0, β1, β2, τ) = {f}")

results = model.estimate_parameters('gradient_descent', t,  starting_points=[0, 0, 0, 1], step_sizes=[0.1, 0.1, 0.1, 0.1])
for instance, result in results.items():
    print(f"Instance: {instance}")
    print(f"Optimal Solution: {result['Optimal Solution']}")
    print(f"Optimal Value: {result['Optimal Value']}")
    print(f"Iterations: {result['Iterations']}")
    print()

model.plot_curve()
'''