import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class NelsonSiegelModel:
    """
    Class to handle the Nelson-Siegel model and its parameter estimation.

    Attributes:
    - beta0, beta1, beta2, tau: float
        These are the parameters of the Nelson-Siegel model.
    - yields: numpy.ndarray
        Array of historical yield data for various securities.
    """

    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float, yields: np.ndarray):
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
        self.tau = tau
        self.yields = yields

    def compute_R(self, t: float) -> float:
        """
        Computes the R(t) value using the Nelson-Siegel model.

        Parameters:
        - t: float
            The time parameter for which R(t) is to be computed.

        Returns:
        - float:
            The computed R(t) value.
        """

        return self.beta0 + self.beta1 * (1 - np.exp(-t / self.tau)) / (t / self.tau) + self.beta2 * ((1 - np.exp(-t / self.tau)) / (t / self.tau) - np.exp(-t / self.tau))

    def compute_f(self, t: float) -> float:
        """
        Computes the f(β0, β1, β2, τ) value using the Nelson-Siegel model.

        Returns:
        - float:
            The computed f(β0, β1, β2, τ) value.
        """

        residuals = self.yields - np.array([self.compute_R(t) for t in range(1, len(self.yields) + 1)])
        return np.sum(residuals ** 2)

    def estimate_parameters(self, method: str = 'gradient_descent', starting_points: list = [0, 0, 0, 1], step_sizes: list = [0.1, 0.1, 0.1, 0.1]) -> dict:
        """
        Estimates the parameters of the Nelson-Siegel model using the specified optimization method.

        Parameters:
        - method: str (default: 'gradient_descent')
            The optimization method to use. Can be 'gradient_descent' or 'newton'.
        - starting_points: list (default: [0, 0, 0, 1])
            List of starting points for the optimization algorithm.
        - step_sizes: list (default: [0.1, 0.1, 0.1, 0.1])
            List of step sizes for the optimization algorithm.

        Returns:
        - dict:
            A dictionary containing the optimal solutions, optimal values, and number of iterations or running time for each instance and algorithm.
        """

        if method == 'gradient_descent':
            results = {}
            for i, start_point in enumerate(starting_points):
                res = minimize(self.compute_f, starting_points, method='BFGS', jac=False, options={'gtol': 1e-6, 'disp': True})
                results[f'Instance {i+1}'] = {'Optimal Solution': res.x, 'Optimal Value': res.fun, 'Iterations': res.nit}
            return results
        elif method == 'newton':
            results = {}
            for i, start_point in enumerate(starting_points):
                res = minimize(self.compute_f, starting_points, method='Newton-CG', jac=False, options={'xtol': 1e-6, 'disp': True})
                results[f'Instance {i+1}'] = {'Optimal Solution': res.x, 'Optimal Value': res.fun, 'Iterations': res.nit}
            return results
        else:
            raise ValueError("Invalid optimization method. Please choose either 'gradient_descent' or 'newton'.")

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

# Historical yield data
yields = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])

# Creating an instance of NelsonSiegelModel
model = NelsonSiegelModel(0.05, -0.02, -0.03, 1, yields)

# Computing R(t) for a specific time
t = 3
R = model.compute_R(t)
print(f"R({t}) = {R}")

# Computing f(β0, β1, β2, τ)
f = model.compute_f(t)
print(f"f(β0, β1, β2, τ) = {f}")

# Estimating parameters using gradient descent method
results = model.estimate_parameters(method='gradient_descent', starting_points=[0, 0, 0, 1], step_sizes=[0.1, 0.1, 0.1, 0.1])
for instance, result in results.items():
    print(f"Instance: {instance}")
    print(f"Optimal Solution: {result['Optimal Solution']}")
    print(f"Optimal Value: {result['Optimal Value']}")
    print(f"Iterations: {result['Iterations']}")
    print()

# Plotting the curve (t, R(t))
model.plot_curve()