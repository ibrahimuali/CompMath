import numpy as np
import matplotlib.pyplot as plt

class NelsonSiegelModel:
    """
    Class to handle the Nelson-Siegel model and compute R(t) and f(β0, β1, β2, τ).

    Attributes:
    - beta0, beta1, beta2, tau: float
        These are the parameters of the Nelson-Siegel model.
    """

    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float):
        """
        Constructor to instantiate the NelsonSiegelModel class.

        Parameters:
        - beta0, beta1, beta2, tau: float
            Parameters of the Nelson-Siegel model.
        """

        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

    def compute_R(self, t: float):
        """
        Computes R(t) using the Nelson-Siegel model.

        Parameters:
        - t: float
            The time parameter for which R(t) is to be computed.

        Returns:
        - float:
            The computed value of R(t).
        """

        R = self.beta0 + self.beta1 * (1 - np.exp(-t / self.tau)) / (t / self.tau) + self.beta2 * ((1 - np.exp(-t / self.tau)) / (t / self.tau) - np.exp(-t / self.tau))
        return R

    def compute_f(self):
        """
        Computes f(β0, β1, β2, τ) using the Nelson-Siegel model.

        Returns:
        - float:
            The computed value of f(β0, β1, β2, τ).
        """

        f = self.beta0 + self.beta1 + self.beta2 + self.tau
        return f

def gradient_descent_approx_line_search(f, initial_params, step_size, max_iterations):
    """
    Performs gradient descent with approximate line search to minimize the objective function f.

    Parameters:
    - f: function
        The objective function to be minimized.
    - initial_params: list
        The initial parameter values.
    - step_size: float
        The step size for each iteration.
    - max_iterations: int
        The maximum number of iterations to perform.

    Returns:
    - list:
        The optimal parameter values.
    - float:
        The optimal value of the objective function.
    - int:
        The number of iterations performed.
    """

    params = np.array(initial_params)
    iterations = 0

    while iterations < max_iterations:
        gradient = np.gradient(f(params))
        new_params = params - step_size * gradient

        if f(new_params) < f(params):
            params = new_params
        else:
            break

        iterations += 1

    return params, f(params), iterations

def newtons_method(f, initial_params, max_iterations):
    """
    Performs Newton's method to minimize the objective function f.

    Parameters:
    - f: function
        The objective function to be minimized.
    - initial_params: list
        The initial parameter values.
    - max_iterations: int
        The maximum number of iterations to perform.

    Returns:
    - list:
        The optimal parameter values.
    - float:
        The optimal value of the objective function.
    - int:
        The number of iterations performed.
    """

    params = np.array(initial_params)
    iterations = 0

    while iterations < max_iterations:
        gradient = np.gradient(f(params))
        hessian = np.gradient(gradient)
        new_params = params - np.linalg.inv(hessian) @ gradient

        if f(new_params) < f(params):
            params = new_params
        else:
            break

        iterations += 1

    return params, f(params), iterations

# Example usage:

# Example 1: Computing R(t) and f(β0, β1, β2, τ)
model = NelsonSiegelModel(0.1, 0.2, 0.3, 0.4)
t_value = 0.5
R_value = model.compute_R(t_value)
f_value = model.compute_f()
print(f"For the Nelson-Siegel model with parameters β0={model.beta0}, β1={model.beta1}, β2={model.beta2}, τ={model.tau}:")
print(f"R({t_value}) = {R_value}")
print(f"f(β0, β1, β2, τ) = {f_value}")

# Example 2: Gradient descent with approximate line search
def objective_function(params):
    return params[0]**2 + params[1]**2

initial_params = [1, 1]
step_size = 0.1
max_iterations = 100
optimal_params, optimal_value, iterations = gradient_descent_approx_line_search(objective_function, initial_params, step_size, max_iterations)
print(f"Optimal parameters: {optimal_params}")
print(f"Optimal value: {optimal_value}")
print(f"Number of iterations: {iterations}")

# Example 3: Newton's method
optimal_params, optimal_value, iterations = newtons_method(objective_function, initial_params, max_iterations)
print(f"Optimal parameters: {optimal_params}")
print(f"Optimal value: {optimal_value}")
print(f"Number of iterations: {iterations}")

# Example 4: Plotting the curve (t, R(t))
t_values = np.linspace(0, 1, 100)
R_values = [model.compute_R(t) for t in t_values]

plt.plot(t_values, R_values)
plt.xlabel('t')
plt.ylabel('R(t)')
plt.title('Nelson-Siegel Model')
plt.show()