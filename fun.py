import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def join_df_date(df1, df2, df3, df4, df5, df6, maturity1, maturity2, maturity3, maturity4, maturity5, maturity6):
    '''
    Join two Dataframe with a column for Price and other one for Maturity for each month.
    
    Return: 
    - all_df: list
        It is a list of Dataframe of each month
    '''
    all_df = []
    data = {'Maturity': [maturity1, maturity2, maturity3, maturity4, maturity5, maturity6]}
    for i in range(len(df1)):
        price = {'Yield': [df1['Price'].iloc[i], df2['Price'].iloc[i], df3['Price'].iloc[i], df4['Price'].iloc[i], df5['Price'].iloc[i], df6['Price'].iloc[i]]}
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
    
def plot_curve(time, yields, R, method, folder_name, date):
    """
    Plots the curve (t, R(t)) using the computed R(t) values and the historical yield data.
    """
    plot_folder = os.path.join(folder_name, method)
    os.makedirs(plot_folder, exist_ok=True)

    plot_folder = os.path.join(plot_folder, 'Plot')
    os.makedirs(plot_folder, exist_ok=True)

    date_without_time = date.strftime('%Y-%m-%d')

    plt.plot(time, yields, label='Historical Yield Data')
    plt.plot(time, R, label=f'Nelson-Siegel Curve [{method}]')
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.title(f'Nelson-Siegel Curve vs Historical Yield Data [{method}]')
    plt.legend()
    plt.savefig(os.path.join(plot_folder, f'{folder_name}-{date_without_time}.png'), dpi=300)
    plt.close()