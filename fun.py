import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import matplotlib.ticker as mtick
from scipy.linalg import solve, eigh
from scipy.optimize import approx_fprime
import os

def clear_df(df):
    '''
    Drop useless columns and set date as index

    Return: DataFrame
    '''

    columns_to_drop = ['Open', 'High', 'Low', 'Change %']
    df.drop(columns=columns_to_drop, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Price'] = (df['Price']/100)
    df = df[::-1]
    return df


def join_df_date(df1, df2, df3, df4, df5, maturity1, maturity2, maturity3, maturity4, maturity5):
    '''
    Join two Dataframe with a column for Price and other one for Maturity for each month.
    
    Return: 
    - all_df: list
        It is a list of Dataframe of each month
    '''
    all_df = []
    data = {'Maturity': [maturity1, maturity2, maturity3, maturity4, maturity5]}
    for i in range(len(df1)):
        price = {'Yield': [df1['Price'].iloc[i], df2['Price'].iloc[i], df3['Price'].iloc[i], df4['Price'].iloc[i], df5['Price'].iloc[i]]}
        df_new = pd.DataFrame({**data, **price})
        all_df.append(df_new)
    return all_df


def compute_R(time, params_NS=None, params_NSS=None):
    """
    Computes the R(t) value using the Nelson-Siegel model.

    Parameters:
    - time: column of dataframe
        The time parameter for which R(t) is to be computed.
    - params: array
        Parameters of the Nelson-Siegel model
    
    Returns:
    - float:
        The computed R(t) value.
    """
    if params_NS is not None:
        beta0, beta1, beta2, tau = params_NS
        f1 = (1 - np.exp(-time / tau)) / (time / tau)
        f2 = (1 - np.exp(-time / tau)) / (time / tau) - np.exp(-time / tau)
        return beta0 + beta1 * f1 + beta2 * f2
    else:
        beta0, beta1, beta2, beta3, tau, tau2 = params_NSS
        f1 = (1 - np.exp(-time / tau)) / (time / tau)
        f2 = (1 - np.exp(-time / tau)) / (time / tau) - np.exp(-time / tau)
        f3 = (1 - np.exp(-time / tau2)) / (time / tau2) - np.exp(-time / tau2)
        return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3
    

def compute_f(yields, time, params_NS=None, params_NSS=None):
        """
        Computes the f(β0, β1, β2, τ) or f(β0, β1, β2, β3, τ_0, τ_1) value using the Nelson-Siegel model.

        Returns:
        - float:
            The computed f(β0, β1, β2, τ) or f(β0, β1, β2, β3, τ_0, τ_1) value.
        """
        if params_NS is not None:
            residuals = yields - compute_R(time, params_NS=params_NS)
        else:
             residuals = yields - compute_R(time, params_NSS=params_NSS)
        return np.sum(residuals**2)
    

def plot_curve(time, yields, R, country, model, method, date):
    """
    Plots the curve (t, R(t)) using the computed R(t) values and the historical yield data.
    """
    folders = [country, model, method, 'Plot']
    plot_folder = os.path.join(*folders)
    os.makedirs(plot_folder, exist_ok=True)

    date_without_time = date.strftime('%Y-%m-%d')

     # Begin plotting     
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")  # Set the plot background color to white
    fig.patch.set_facecolor('white')

    # Plotting the yield data and the Nelson-Siegel model predictions
    ax.plot(time, np.array(yields) * 100, 'o-', color="blue", label='Actual Yield')  # Actual yields
    ax.plot(time, np.array(R) * 100, 'o-', color="orange", label=f'{method} Predictions')  # Nelson-Siegel predictions
    
    # Convert yields to percentages and find the min and max
    yield_percentages = np.array(yields) * 100
    min_yield = np.floor(min(yield_percentages) * 2) / 2  # Round down to the nearest 0.5%
    max_yield = np.ceil(max(yield_percentages) * 2) / 2  # Round up to the nearest 0.5%

    # Create a range of ticks from min to max yield, with steps of 0.5%
    y_ticks = np.arange(min_yield, max_yield + 0.5, 0.3)
    
    # Formatting the plot
    ax.set_title(f'{method} - Fitted Yield Curve', fontsize=12)
    ax.set_xlabel('Period', fontsize=10)
    ax.set_ylabel('Interest', fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_ticks(np.arange(1, max(time)+1, 1))
    ax.yaxis.set_ticks(y_ticks)
    ax.legend(loc="lower right", title="Legend")
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
   
    plt.savefig(os.path.join(plot_folder, f'{country}-{date_without_time}.png'))
    plt.close(fig)
     

def excel(list, country, model, method, name_data):
    
    folders = [country, model, method, 'Tables']
    excel_folder = os.path.join(*folders)
    os.makedirs(excel_folder, exist_ok=True)
    excel_file = os.path.join(excel_folder, f'{country}.xlsx')

    df = pd.DataFrame(list)
    
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=name_data, index=False)
    else:
        df.to_excel(excel_file, sheet_name=name_data)


def newton_method(f, x_0, N, damping_factor = 0.5, eps=1e-6):
    '''
    This function is used to approximate the function values using Newton's method.
    In our case we will use it to Approximate β0, β1, β2, τ for the Nelson-Siegel (NS) model.
    
    Parameters:
    - f(x): The sum of squared difference
        The f(β0, β1, β2, τ) we use to asses how well our model is approximated
    - x_0: Starting points
        The starting points for our Model, it is array of 4 values - β0, β1, β2, τ 
    - N: Number of iterations

    Returns: 
    - Array: Approximated x-values 
    - Floats: minimized f(x)-value
    '''
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
    '''
    Check if the Hessian Matrix is PSD.
    
    Parameters:
    - matrix:
        Hessian Matrix

    Return: all non-negative eigenvalues
    '''
    
    # Compute the eigenvalues
    eigenvalues = eigh(matrix, eigvals_only = True)
    return np.all(eigenvalues > tol)


def approx_hessian(x, f):
    '''
    Approximate values of Hessian Matrix then to use it in Newton-Raphson method
    
    Parameters:
    - x: x-values for the function, starting with starting points 
    - f(x): the objective function
    
    Return: Hessian Matrix
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


def gradient_descent(f, x_0, alpha_0, apx_LS, N, eps = 1e-4):
    '''
    This function is used to approximate the function values using Gradient Descent method.
    In our case we will use it to Approximate β0, β1, β2, τ for the Nelson-Siegel (NS) model.
    
    Parameters:
    - f(x): The sum of squared difference
        The f(β0, β1, β2, τ) we use to asses how well our model is approximated
    - x_0: Starting points
        The starting points for our Model, it is array of 4 values - β0, β1, β2, τ 
    - alpha_0: 
        Learning rate or step size for descent
    - apx_LS: Approximate Line Search
        The function to check if we satisfy Armijo-Goldstein condition
    - N: Number of iterations

    Returns: 
    - Array: Approximated x-values 
    - Floats: minimized f(x) 
    '''
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

def apx_line_search(f, x, d, alpha_0, c = 0.2, t = 0.8):
    '''
    This function checks if Armijo-Goldstein condition is satisfied to ensure
    a sufficient decrease in the objective function.
    
    Parameters:
    - f(x): The sum of squared difference
        The f(β0, β1, β2, τ) we use to asses how well our model is approximated
    - x: Function Points
        Array of 4 values - β0, β1, β2, τ, will update in each ittiration
    - d: Gradient Descent
        Descent direction
    - alpha_0: Steps or Learning Rate
        To check if objective functiong decreasing with sufficient speed
    - t: Scaling Factor 
        0 < t < 1
    - c: constant 0 < c < 1
    - N: Number of iterations
    
    Return: 
    - Floats: new alpha - learning rate
    '''
    alpha = alpha_0
    f_x = f(x)
    
    def phi(a):
        return f(x+ a*d)
    
    phi_prime = approx_fprime(0, phi)
    
    while  phi(alpha) > f_x +c*alpha*phi_prime:
        alpha *= t
        
    return alpha

def compute_f_lm(yields, time, params_NS=None, params_NSS=None):
    """
    Computes the residuals using the Nelson-Siegel model.

    Returns:
    - np.array:
        The residuals.
    """
    if params_NS is not None:
        residuals = yields - compute_R(time, params_NS=params_NS)
    else:
        residuals = yields - compute_R(time, params_NSS=params_NSS)
    return residuals
